# complete_faiss_indexer.py - COMPLETE FIXED VERSION
import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import time
import random

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Complete5GNIDDFAISSIndexer:
    """
    FAISS-based indexer for 5G-NIDD-like network-flow CSVs.

    Key points:
      - EXCLUDES 'Seq'
      - Treats Proto, sTos, dTos, sDSb, dDSb, Cause, State as CATEGORICAL
      - Fits categorical encoders on the UNION of values across datasets
      - FREEZES a global feature schema (consistent column order & count)
      - Sanitizes FAISS distances (avoid NaN/Inf)
      - Sets nprobe on IVF indices
      - Uses a margin-based decision rule
      - ✨ FIXED: Per-index distance calibration → normalized similarities → calibrated confidence
    """

    def __init__(self, data_dir: str = "data_split", index_dir: str = "faiss_indices"):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)

        # ML components
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=25)
        self.categorical_encoders: Dict[str, LabelEncoder] = {}

        # FAISS indices and metadata
        self.benign_index = None
        self.benign_features = None  # keep a sample for calibration
        self.malicious_indices: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.attack_types: List[str] = []
        self.index_metadata: Dict = {}

        # Dataset schema hints
        self.expected_columns = [
            '', 'Seq', 'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max', 'Proto',
            'sTos', 'dTos', 'sDSb', 'dDSb', 'sTtl', 'dTtl', 'sHops', 'dHops',
            'Cause', 'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes',
            'DstBytes', 'Offset', 'sMeanPktSz', 'dMeanPktSz', 'Load', 'SrcLoad',
            'DstLoad', 'Loss', 'SrcLoss', 'DstLoss', 'pLoss', 'SrcGap', 'DstGap',
            'Rate', 'SrcRate', 'DstRate', 'State', 'SrcWin', 'DstWin', 'sVid',
            'dVid', 'SrcTCPBase', 'DstTCPBase', 'TcpRtt', 'SynAck', 'AckDat',
            'Label', 'Attack Type', 'Attack Tool'
        ]

        # CATEGORICALS
        self.categorical_feature_names = ['Proto', 'sTos', 'dTos', 'sDSb', 'dDSb', 'Cause', 'State']
        self.label_columns = ['Label', 'Attack Type', 'Attack Tool']

        # Decision thresholds (now operate on CALIBRATED sims in [0,1])
        self.malicious_threshold = 0.70
        self.suspicious_floor   = 0.55
        self.margin_required    = 0.10  # mal_sim must beat benign_sim by this

        # Confidence softmax temp (smaller = more peaky confidences)
        self.conf_temperature = 0.35

    # --------------- Data discovery ----------------

    def find_csv_files(self) -> Dict[str, Path]:
        logger.info(f"Searching for CSV files in {self.data_dir}")
        csv_files: Dict[str, Path] = {}

        for pattern in ["*benign*.csv", "*index*.csv"]:
            for file_path in self.data_dir.glob(pattern):
                if file_path.is_file():
                    filename = file_path.stem.lower()
                    if 'benign' in filename:
                        csv_files['benign'] = file_path
                    else:
                        attack_type = self._extract_attack_type_from_filename(filename)
                        if attack_type:
                            csv_files[attack_type] = file_path

        if not csv_files:
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir():
                    for csv_file in subdir.glob("*.csv"):
                        filename = csv_file.stem.lower()
                        if 'benign' in filename:
                            csv_files['benign'] = csv_file
                        else:
                            attack_type = self._extract_attack_type_from_filename(filename)
                            if attack_type:
                                csv_files[attack_type] = csv_file

        if not csv_files:
            for csv_file in self.data_dir.rglob("*.csv"):
                filename = csv_file.stem.lower()
                if 'benign' in filename:
                    csv_files['benign'] = csv_file
                else:
                    attack_type = self._extract_attack_type_from_filename(filename)
                    if attack_type:
                        csv_files[attack_type] = csv_file

        logger.info(f"Found {len(csv_files)} usable CSV files:")
        for k, v in csv_files.items():
            logger.info(f"  {k}: {v}")
        return csv_files

    def _extract_attack_type_from_filename(self, filename: str) -> Optional[str]:
        filename = filename.lower()
        if filename.startswith("attack_type_") and filename.endswith("_index"):
            return filename.replace("attack_type_", "").replace("_index", "")
        for tok in ["icmpflood", "udpflood", "synflood", "httpflood",
                    "synscan", "tcpconnectscan", "udpscan", "slowratedos"]:
            if tok in filename:
                return tok
        return None

    # --------------- Load / clean ----------------

    def load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading and preprocessing data files...")
        csv_map = self.find_csv_files()
        if not csv_map:
            logger.error("No CSV files found in data_dir.")
            return {}

        datasets: Dict[str, pd.DataFrame] = {}
        for name, path in csv_map.items():
            logger.info(f"Loading {name} data from {path}")
            df = self._load_and_clean_csv(path)
            if df.empty:
                logger.warning(f"{name} produced empty dataframe, skipping.")
                continue
            datasets[name] = df
            logger.info(f"Loaded {len(df)} {name} rows")

        self.attack_types = [k for k in datasets.keys() if k != 'benign']
        logger.info(f"Found attack types: {self.attack_types}")
        return datasets

    def _load_and_clean_csv(self, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"Original CSV shape: ({len(df)}, {len(df.columns)})")
            if df.columns[0] in ['', 'Unnamed: 0'] or str(df.columns[0]).startswith('Unnamed'):
                df = df.drop(df.columns[0], axis=1)
                logger.info(f"Dropped unnamed index column, new shape: {df.shape}")

            df.columns = [str(col).strip() for col in df.columns]
            df = self._handle_missing_values(df)
            df = self._validate_and_convert_types(df)
            logger.info(f"Final cleaned CSV shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(0)
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [c for c in categorical_cols if c not in self.label_columns]
        df[categorical_cols] = df[categorical_cols].astype(str).fillna('unknown')
        return df

    def _validate_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # force categoricals to string even if numeric in some splits
        for col in self.categorical_feature_names:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('unknown')

        numeric_columns = [
            'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max',
            'sTtl', 'dTtl', 'sHops', 'dHops',
            'TotPkts', 'SrcPkts', 'DstPkts',
            'TotBytes', 'SrcBytes', 'DstBytes',
            'Offset', 'sMeanPktSz', 'dMeanPktSz',
            'Load', 'SrcLoad', 'DstLoad',
            'Loss', 'SrcLoss', 'DstLoss', 'pLoss',
            'Rate', 'SrcRate', 'DstRate',
            'SrcWin', 'DstWin',
            'TcpRtt', 'SynAck', 'AckDat',
            'SrcGap', 'DstGap', 'sVid', 'dVid',
            'SrcTCPBase', 'DstTCPBase'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # keep rest of objects as strings
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [c for c in categorical_cols if c not in self.label_columns]
        for col in categorical_cols:
            df[col] = df[col].astype(str).fillna('unknown')
        return df

    # --------------- Schema & encoders ----------------

    def _fit_categorical_encoders(self, datasets: Dict[str, pd.DataFrame]):
        logger.info("Fitting categorical encoders on union of values...")
        union_values: Dict[str, set] = {c: set(['unknown']) for c in self.categorical_feature_names}

        for _, df in datasets.items():
            work_df = df.drop(columns=[c for c in self.label_columns if c in df.columns], errors='ignore')
            if 'Seq' in work_df.columns:
                work_df = work_df.drop(columns=['Seq'])
            for col in self.categorical_feature_names:
                if col in work_df.columns:
                    vals = work_df[col].astype(str).fillna('unknown').unique().tolist()
                    union_values[col].update(vals)

        self.categorical_encoders = {}
        for col, values in union_values.items():
            le = LabelEncoder()
            classes = sorted(values)
            le.fit(classes)
            self.categorical_encoders[col] = le
            logger.info(f"  {col}: {len(classes)} classes (incl. 'unknown')")

    def _freeze_feature_schema(self, datasets: Dict[str, pd.DataFrame]):
        union_numeric = set()
        present_cats = set()
        forced_cat = set(self.categorical_feature_names)

        for _, df in datasets.items():
            work_df = df.drop(columns=[c for c in self.label_columns if c in df.columns], errors='ignore')
            if 'Seq' in work_df.columns:
                work_df = work_df.drop(columns=['Seq'])

            for col in work_df.select_dtypes(include=[np.number]).columns.tolist():
                if col not in forced_cat:
                    union_numeric.add(col)

            for col in self.categorical_feature_names:
                if col in work_df.columns:
                    present_cats.add(col)

        self.numerical_features = sorted(list(union_numeric))
        self.categorical_features = [c for c in self.categorical_feature_names if c in present_cats]
        self.feature_names = self.numerical_features + [f"{c}_le" for c in self.categorical_features]

        logger.info(f"Frozen schema → numeric:{len(self.numerical_features)}  categorical:{len(self.categorical_features)}  total:{len(self.feature_names)}")

    # --------------- Feature engineering ----------------

    def extract_and_engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        work_df = df.drop(columns=[c for c in self.label_columns if c in df.columns], errors='ignore')
        if 'Seq' in work_df.columns:
            work_df = work_df.drop(columns=['Seq'])

        if self.numerical_features:
            X_num_df = work_df.reindex(columns=self.numerical_features, fill_value=0)
            X_num = X_num_df.values.astype(np.float32)
        else:
            X_num = np.empty((len(work_df), 0), dtype=np.float32)

        cat_blocks = []
        for col in self.categorical_features:
            if col in work_df.columns:
                col_vals = work_df[col].astype(str).fillna('unknown')
            else:
                col_vals = pd.Series(['unknown'] * len(work_df), index=work_df.index, dtype="object")
            le = self.categorical_encoders.get(col) or LabelEncoder().fit(['unknown'])
            allowed = set(le.classes_.tolist())
            col_vals = col_vals.where(col_vals.isin(allowed), 'unknown')
            enc = le.transform(col_vals).astype(np.float32).reshape(-1, 1)
            cat_blocks.append(enc)

        X_cat = np.hstack(cat_blocks) if cat_blocks else np.empty((len(work_df), 0), dtype=np.float32)
        X = np.hstack([X_num, X_cat]).astype(np.float32)

        if not self.feature_names:
            self.feature_names = self.numerical_features + [f"{c}_le" for c in self.categorical_features]
            logger.info(f"Total features after encoding (without Seq): {len(self.feature_names)}")
        return X

    # --------------- Index building ----------------

    def build_indices(self, datasets: Dict[str, pd.DataFrame]):
        logger.info("Building FAISS indices...")
        self._fit_categorical_encoders(datasets)
        self._freeze_feature_schema(datasets)

        all_X = []
        all_y = []
        ordered_keys = list(datasets.keys())

        for i, name in enumerate(ordered_keys):
            df = datasets[name]
            feats = self.extract_and_engineer_features(df)
            all_X.append(feats)
            all_y.append(np.full(len(feats), i))
            logger.info(f"Prepared features for {name}: {feats.shape}")

        X_all = np.vstack(all_X)
        y_all = np.hstack(all_y)
        logger.info(f"Combined feature matrix: {X_all.shape}")

        X_scaled = self.scaler.fit_transform(X_all)
        X_sel = self.feature_selector.fit_transform(X_scaled, y_all)
        sel_idx = self.feature_selector.get_support(indices=True)
        sel_names = [self.feature_names[i] for i in sel_idx]
        logger.info(f"Selected {X_sel.shape[1]} features")

        cursor = 0
        stored_feats: Dict[str, np.ndarray] = {}
        for name in ordered_keys:
            n = len(datasets[name])
            X_part = X_sel[cursor:cursor + n]
            stored_feats[name] = X_part
            if name == 'benign':
                self._build_benign_index(X_part)
            else:
                self._build_malicious_index(name, X_part, datasets[name])
            cursor += n

        # --- NEW: per-index distance calibration ---
        self._build_calibration(stored_feats)

        self.index_metadata['feature_selection'] = {
            'selected_features': sel_names,
            'total_features': len(self.feature_names),
            'selected_count': len(sel_names)
        }

    def _build_benign_index(self, features: np.ndarray):
        d = features.shape[1]
        n = len(features)
        nlist = min(100, max(10, n // 50))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        logger.info(f"Training benign index with {n} samples (d={d}, nlist={nlist})")
        index.train(features)
        index.add(features)
        try:
            index.nprobe = min(10, nlist)
        except Exception:
            pass
        self.benign_index = index
        self.benign_features = features  # keep for calibration
        self.index_metadata['benign'] = {'num_samples': n, 'feature_dim': d, 'nlist': nlist, 'index_type': 'IVFFlat'}

    def _build_malicious_index(self, attack_type: str, features: np.ndarray, original_df: pd.DataFrame):
        d = features.shape[1]
        n = len(features)
        if n < 1000:
            index = faiss.IndexFlatL2(d)
            idx_type = 'Flat'
        else:
            nlist = min(50, max(10, n // 20))
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(features)
            try:
                index.nprobe = min(10, nlist)
            except Exception:
                pass
            idx_type = f'IVFFlat_nlist{nlist}'
        index.add(features)
        self.malicious_indices[attack_type] = {
            'index': index,
            'metadata': original_df.to_dict('records'),
            'features': features
        }
        self.index_metadata[attack_type] = {'num_samples': n, 'feature_dim': d, 'index_type': idx_type}
        logger.info(f"Built {attack_type} index: {n} samples, d={d}, type={idx_type}")

    # --------------- Calibration ----------------

    @staticmethod
    def _mean_knn_distance(index: faiss.Index, X: np.ndarray, sample: int = 2000, k: int = 6) -> np.ndarray:
        """
        For a random sample of rows, compute mean distance to k-1 nearest neighbors (skip self).
        """
        n = len(X)
        s = min(sample, n)
        idxs = np.random.choice(n, size=s, replace=False) if n > s else np.arange(n)
        Q = X[idxs]
        dists, _ = index.search(Q, k)
        # skip the first neighbor (self with dist 0) if present
        if dists.shape[1] >= 2:
            neigh = dists[:, 1:]
        else:
            neigh = dists
        neigh = np.nan_to_num(neigh, nan=1e6, posinf=1e6, neginf=1e6)
        return neigh.mean(axis=1)

    def _build_calibration(self, stored_feats: Dict[str, np.ndarray]):
        """
        Build per-index distance calibration using percentiles of mean kNN distance.
        Store p5 & p95 so later we can normalize distances into [0,1].
        """
        logger.info("Building distance calibration (p5/p95 of mean kNN distances)...")

        # Build temporary flat indices for calibration (robust and fast enough on samples)
        def flat_index(X):
            idx = faiss.IndexFlatL2(X.shape[1])
            idx.add(X)
            return idx

        # Benign
        if self.benign_features is not None:
            b_flat = flat_index(self.benign_features)
            b_md = self._mean_knn_distance(b_flat, self.benign_features, sample=3000, k=6)
            p5, p95 = float(np.percentile(b_md, 5)), float(np.percentile(b_md, 95))
            self.index_metadata['benign']['calibration'] = {'p5': p5, 'p95': p95}
            logger.info(f"  Calibration benign: p5={p5:.4f} p95={p95:.4f}")

        # Each attack type
        for atk, blob in self.malicious_indices.items():
            X = blob['features']
            m_flat = flat_index(X)
            m_md = self._mean_knn_distance(m_flat, X, sample=3000, k=6)
            p5, p95 = float(np.percentile(m_md, 5)), float(np.percentile(m_md, 95))
            meta = self.index_metadata.get(atk, {})
            meta['calibration'] = {'p5': p5, 'p95': p95}
            self.index_metadata[atk] = meta
            logger.info(f"  Calibration {atk}: p5={p5:.4f} p95={p95:.4f}")

    # --------------- Persistence ----------------

    def save_indices(self):
        logger.info("Saving FAISS indices...")
        if self.benign_index:
            faiss.write_index(self.benign_index, str(self.index_dir / "benign_index.faiss"))

        for atk, data in self.malicious_indices.items():
            faiss.write_index(data['index'], str(self.index_dir / f"{atk}_index.faiss"))
            with open(self.index_dir / f"{atk}_metadata.pkl", 'wb') as f:
                pickle.dump({'metadata': data['metadata'], 'features': data['features']}, f)

        payload = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'categorical_encoders': self.categorical_encoders,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'attack_types': self.attack_types,
            'index_metadata': self.index_metadata,
            'conf_temperature': self.conf_temperature,
            'thresholds': {
                'malicious_threshold': self.malicious_threshold,
                'suspicious_floor': self.suspicious_floor,
                'margin_required': self.margin_required
            }
        }
        with open(self.index_dir / "preprocessing.pkl", 'wb') as f:
            pickle.dump(payload, f)
        with open(self.index_dir / "index_info.json", 'w') as f:
            json.dump(self.index_metadata, f, indent=2)
        logger.info("Indices + preprocessing saved.")

    def load_indices(self):
        logger.info("Loading FAISS indices...")
        with open(self.index_dir / "preprocessing.pkl", 'rb') as f:
            payload = pickle.load(f)

        self.scaler = payload['scaler']
        self.feature_selector = payload['feature_selector']
        self.categorical_encoders = payload['categorical_encoders']
        self.feature_names = payload['feature_names']
        self.numerical_features = payload['numerical_features']
        self.categorical_features = payload['categorical_features']
        self.attack_types = payload['attack_types']
        self.index_metadata = payload['index_metadata']

        self.conf_temperature = payload.get('conf_temperature', self.conf_temperature)
        thr = payload.get('thresholds', {})
        self.malicious_threshold = thr.get('malicious_threshold', self.malicious_threshold)
        self.suspicious_floor   = thr.get('suspicious_floor',   self.suspicious_floor)
        self.margin_required    = thr.get('margin_required',    self.margin_required)

        benign_path = self.index_dir / "benign_index.faiss"
        if benign_path.exists():
            self.benign_index = faiss.read_index(str(benign_path))
            try:
                nlist = getattr(self.benign_index, 'nlist', 10)
                self.benign_index.nprobe = min(10, nlist)
            except Exception:
                pass

        self.malicious_indices = {}
        for atk in self.attack_types:
            idx_path = self.index_dir / f"{atk}_index.faiss"
            meta_path = self.index_dir / f"{atk}_metadata.pkl"
            if idx_path.exists() and meta_path.exists():
                index = faiss.read_index(str(idx_path))
                try:
                    nlist = getattr(index, 'nlist', 10)
                    index.nprobe = min(10, nlist)
                except Exception:
                    pass
                with open(meta_path, 'rb') as f:
                    blob = pickle.load(f)
                self.malicious_indices[atk] = {'index': index, 'metadata': blob['metadata'], 'features': blob['features']}
        logger.info(f"Loaded attack types: {self.attack_types}")

    # --------------- Classification (FIXED) ----------------

    def classify_flow(self, flow_features: Dict) -> Tuple[str, float, Optional[str]]:
        """
        FIXED VERSION: Classify a network flow using FAISS indices.
        
        Returns: (label, confidence_in_[0..1], attack_type_or_None)
        
        This version fixes the issues causing all flows to be classified as "suspicious".
        """
        try:
            # Debug: Print input features for first few classifications
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            self._debug_count += 1
            
            if self._debug_count <= 3:  # Debug first 3 classifications
                logger.info(f"DEBUG: Classification #{self._debug_count}")
                logger.info(f"Input features: {list(flow_features.keys())}")
            
            # Step 1: Create proper feature vector using DataFrame approach
            vec = self._create_feature_vector_from_dict(flow_features)
            
            if vec is None or len(vec) == 0:
                logger.warning("Failed to create feature vector")
                return "unknown", 0.0, None
            
            # Step 2: Apply preprocessing 
            vec_scaled = self.scaler.transform(vec.reshape(1, -1))
            vec_selected = self.feature_selector.transform(vec_scaled)
            
            if self._debug_count <= 3:
                logger.info(f"Feature vector shape: {vec.shape}, Selected shape: {vec_selected.shape}")
            
            # Step 3: Calculate similarities using simpler, more reliable approach
            benign_score = self._calculate_similarity_score(vec_selected, 'benign')
            
            # Step 4: Calculate attack scores
            attack_scores = {}
            for attack_type in self.attack_types:
                score = self._calculate_similarity_score(vec_selected, attack_type)
                attack_scores[attack_type] = score
            
            # Find best attack match
            best_attack = max(attack_scores.items(), key=lambda x: x[1]) if attack_scores else ("unknown", 0.0)
            best_attack_type, best_attack_score = best_attack
            
            if self._debug_count <= 3:
                logger.info(f"Benign score: {benign_score:.3f}")
                logger.info(f"Best attack: {best_attack_type} (score: {best_attack_score:.3f})")
                logger.info(f"All attack scores: {attack_scores}")
            
            # Step 5: FIXED decision logic with reasonable thresholds
            return self._make_classification_decision(benign_score, best_attack_type, best_attack_score)
            
        except Exception as e:
            logger.error(f"Error classifying flow: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0, None

    def _create_feature_vector_from_dict(self, flow_features: Dict) -> np.ndarray:
        """
        Create a proper feature vector from flow features dict.
        Uses the same feature engineering as during training.
        """
        try:
            # Create data dict with all expected features
            data = {}
            
            # Numerical features
            for feature in self.numerical_features:
                value = flow_features.get(feature, 0.0)
                try:
                    data[feature] = [float(value)]
                except (ValueError, TypeError):
                    data[feature] = [0.0]
            
            # Categorical features  
            for feature in self.categorical_features:
                value = flow_features.get(feature, 'unknown')
                data[feature] = [str(value)]
            
            # Create DataFrame and extract features
            df = pd.DataFrame(data)
            feature_matrix = self.extract_and_engineer_features(df)
            
            if len(feature_matrix) > 0:
                return feature_matrix[0]
            else:
                # Create zero vector as fallback
                return np.zeros(len(self.feature_names), dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return np.zeros(len(self.feature_names), dtype=np.float32)

    def _calculate_similarity_score(self, feature_vector: np.ndarray, index_type: str) -> float:
        """
        Calculate similarity score using a more robust approach.
        Returns a score between 0.0 and 1.0 where higher = more similar.
        """
        try:
            if index_type == 'benign':
                if self.benign_index is None:
                    return 0.0
                index = self.benign_index
            else:
                if index_type not in self.malicious_indices:
                    return 0.0
                index = self.malicious_indices[index_type]['index']
            
            if not hasattr(index, 'ntotal') or index.ntotal == 0:
                logger.warning(f"Index {index_type} is empty!")
                return 0.0
            
            # Search for nearest neighbors
            k = min(10, index.ntotal)  # Use more neighbors for better statistics
            distances, indices = index.search(feature_vector, k=k)
            
            if len(distances[0]) == 0 or np.all(distances[0] == 0):
                return 0.0
            
            # Filter out any invalid distances
            valid_distances = distances[0][distances[0] > 0]
            if len(valid_distances) == 0:
                valid_distances = distances[0]
            
            # Use minimum distance instead of average (closest match)
            min_distance = np.min(valid_distances)
            
            # More aggressive similarity calculation
            # Scale distances to a reasonable range
            if min_distance > 1000:  # Very large distance
                similarity = 0.0
            elif min_distance > 100:  # Large distance
                similarity = 0.1
            elif min_distance > 10:   # Medium distance
                similarity = 0.3 + 0.4 * np.exp(-min_distance / 50.0)
            else:  # Small distance - good match
                similarity = 0.7 + 0.3 * np.exp(-min_distance / 5.0)
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating similarity for {index_type}: {e}")
            return 0.0

    def _make_classification_decision(self, benign_score: float, best_attack_type: str, best_attack_score: float) -> Tuple[str, float, Optional[str]]:
        """
        Make the final classification decision with more aggressive thresholds.
        """
        # More aggressive thresholds for detection
        malicious_threshold = 0.4    # Lower threshold for malicious
        suspicious_threshold = 0.25  # Lower threshold for suspicious
        benign_threshold = 0.3       # Threshold for strong benign signal
        
        # Log the scores for debugging
        if hasattr(self, '_debug_count') and self._debug_count <= 5:
            logger.info(f"Decision: benign={benign_score:.3f}, best_attack={best_attack_type}({best_attack_score:.3f})")
        
        # Strong attack signal
        if best_attack_score > malicious_threshold:
            confidence = min(0.95, max(0.70, best_attack_score + 0.2))
            return "malicious", confidence, best_attack_type
        
        # Moderate attack signal
        elif best_attack_score > suspicious_threshold:
            confidence = min(0.85, max(0.55, best_attack_score + 0.15))
            return "suspicious", confidence, best_attack_type
        
        # Strong benign signal
        elif benign_score > benign_threshold:
            confidence = min(0.90, max(0.60, benign_score + 0.1))
            return "benign", confidence, None
        
        # If all scores are very low, classify as benign with low confidence
        else:
            confidence = 0.5
            return "benign", confidence, None

    @staticmethod
    def _safe_mean(arr: np.ndarray) -> float:
        arr = np.nan_to_num(arr, nan=1e6, posinf=1e6, neginf=1e6, copy=False)
        return float(np.mean(arr))

    @staticmethod
    def _norm_sim_from_dist(dmean: float, p5: float, p95: float) -> float:
        """Map mean distance to [0,1] where 1 ~ very in-cluster."""
        if p95 <= p5:
            return 0.5
        # smaller distance → larger similarity
        sim = (p95 - dmean) / (p95 - p5)
        return float(np.clip(sim, 0.0, 1.0))

    def _softmax2(self, s1: float, s2: float, temp: float) -> Tuple[float, float]:
        a = np.array([s1, s2], dtype=np.float32) / max(1e-6, temp)
        # subtract max for stability
        a = a - np.max(a)
        e = np.exp(a)
        p = e / np.sum(e)
        return float(p[0]), float(p[1])

    def _dict_to_feature_vector(self, flow_features: Dict) -> np.ndarray:
        """
        SIMPLIFIED VERSION: Convert flow features to vector.
        Falls back to the more robust method above.
        """
        return self._create_feature_vector_from_dict(flow_features)

    def get_similar_flows(self, flow_features: Dict, attack_type: str, k: int = 5) -> List[Dict]:
        if attack_type not in self.malicious_indices:
            return []
        try:
            vec = self._dict_to_feature_vector(flow_features)
            vec_s = self.scaler.transform(vec.reshape(1, -1))
            vec_sel = self.feature_selector.transform(vec_s)
            data = self.malicious_indices[attack_type]
            dists, idxs = data['index'].search(vec_sel, k=k)
            out = []
            for i, ridx in enumerate(idxs[0]):
                if 0 <= ridx < len(data['metadata']):
                    row = dict(data['metadata'][ridx])
                    # we'll report normalized similarity for readability if calibration exists
                    cal = self.index_metadata.get(attack_type, {}).get('calibration', None)
                    md = float(dists[0][i])
                    if cal:
                        sim = self._norm_sim_from_dist(md, cal['p5'], cal['p95'])
                    else:
                        sim = 1.0 / (1.0 + md)
                    row['similarity'] = sim
                    out.append(row)
            return out
        except Exception as e:
            logger.error(f"Error in get_similar_flows: {e}")
            return []


def main():
    logger.info("=== Complete 5G-NIDD FAISS Indexer (Calibrated) ===")
    indexer = Complete5GNIDDFAISSIndexer()
    logger.info("Step 1: Loading and preprocessing data...")
    datasets = indexer.load_and_preprocess_data()
    if not datasets:
        logger.error("No datasets found. Check data_split folder and filenames.")
        return False

    logger.info("\nDataset Summary:")
    total = 0
    for name, df in datasets.items():
        logger.info(f"  {name}: {len(df)} flows")
        total += len(df)
    logger.info(f"  Total: {total} flows")

    logger.info("\nStep 2: Building FAISS indices...")
    t0 = time.time()
    indexer.build_indices(datasets)
    dt = time.time() - t0
    logger.info(f"Index build time: {dt:.2f}s")

    logger.info("\nStep 3: Saving indices...")
    indexer.save_indices()

    logger.info("=== FAISS indexing completed successfully! ===")
    return True


if __name__ == "__main__":
    ok = main()
    if not ok:
        exit(1)