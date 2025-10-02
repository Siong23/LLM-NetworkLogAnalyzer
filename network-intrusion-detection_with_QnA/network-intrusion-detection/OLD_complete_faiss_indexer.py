import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Complete5GNIDDFAISSIndexer:
    """
    Complete FAISS-based indexer for the 5G-NIDD network flow dataset.
    Handles the actual CSV structure with proper feature engineering.
    """
    
    def __init__(self, data_dir: str = "data_split", index_dir: str = "faiss_indices"):
        """
        Initialize the FAISS indexer.
        
        Args:
            data_dir: Directory containing split CSV files
            index_dir: Directory to save FAISS indices
        """
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=25)  # Top 25 features as per paper
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}
        
        # FAISS indices
        self.benign_index = None
        self.malicious_indices = {}
        
        # Metadata
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.attack_types = []
        self.index_metadata = {}
        
        # Dataset schema based on actual CSV structure
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
        
        # Define categorical and numerical features
        self.categorical_feature_names = ['Proto', 'sDSb', 'dDSb', 'Cause', 'State']
        self.label_columns = ['Label', 'Attack Type', 'Attack Tool']
        
        # Classification thresholds
        self.benign_threshold = 0.85
        self.malicious_threshold = 0.75
        
        # Feature importance from the paper (these are the top features mentioned)
        self.top_features_from_paper = [
            'Seq', 'Offset', 'sTtl', 'tcp', 'AckDat', 'sMeanPktSz', 'sHops', 
            'Mean', 'dTtl', 'SrcBytes', 'TotBytes', 'dMeanPktSz', 'TcpRtt'
        ]
    
    def find_csv_files(self) -> Dict[str, Path]:
        """
        Find CSV files in the data directory structure.
        Handles both flat structure and subfolder structure.
        
        Returns:
            Dictionary mapping attack types to file paths
        """
        logger.info(f"Searching for CSV files in {self.data_dir}")
        csv_files = {}
        
        # First, try to find files in the root data directory
        for pattern in ["*benign*.csv", "*index*.csv"]:
            for file_path in self.data_dir.glob(pattern):
                if file_path.is_file():
                    logger.info(f"Found file in root: {file_path}")
                    # Extract type from filename
                    filename = file_path.stem.lower()
                    if 'benign' in filename:
                        csv_files['benign'] = file_path
                    else:
                        # Extract attack type from filename
                        attack_type = self._extract_attack_type_from_filename(filename)
                        if attack_type:
                            csv_files[attack_type] = file_path
        
        # If no files found in root, search in subdirectories
        if not csv_files:
            logger.info("No files found in root, searching subdirectories...")
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir():
                    logger.info(f"Searching in subdirectory: {subdir}")
                    for csv_file in subdir.glob("*.csv"):
                        logger.info(f"Found file: {csv_file}")
                        filename = csv_file.stem.lower()
                        if 'benign' in filename:
                            csv_files['benign'] = csv_file
                        else:
                            attack_type = self._extract_attack_type_from_filename(filename)
                            if attack_type:
                                csv_files[attack_type] = csv_file
        
        # Final check - try common patterns
        if not csv_files:
            logger.info("Still no files found, trying broader search...")
            all_csv_files = list(self.data_dir.rglob("*.csv"))
            logger.info(f"Found {len(all_csv_files)} CSV files total:")
            for csv_file in all_csv_files:
                logger.info(f"  - {csv_file}")
                filename = csv_file.stem.lower()
                if 'benign' in filename:
                    csv_files['benign'] = csv_file
                else:
                    attack_type = self._extract_attack_type_from_filename(filename)
                    if attack_type:
                        csv_files[attack_type] = csv_file
        
        logger.info(f"Found {len(csv_files)} usable CSV files:")
        for attack_type, file_path in csv_files.items():
            logger.info(f"  {attack_type}: {file_path}")
        
        return csv_files
    
    def _extract_attack_type_from_filename(self, filename: str) -> Optional[str]:
        """Extract attack type from filename."""
        filename = filename.lower()
        
        # Common attack type patterns from the split_summary.json
        attack_patterns = {
            'icmpflood': 'icmpflood',
            'icmp_flood': 'icmpflood',
            'udpflood': 'udpflood', 
            'udp_flood': 'udpflood',
            'synflood': 'synflood',
            'syn_flood': 'synflood',
            'httpflood': 'httpflood',
            'http_flood': 'httpflood',
            'synscan': 'synscan',
            'syn_scan': 'synscan',
            'tcpconnectscan': 'tcpconnectscan',
            'tcp_connect_scan': 'tcpconnectscan',
            'udpscan': 'udpscan',
            'udp_scan': 'udpscan',
            'slowrateDos': 'slowrateDos',
            'slowrate_dos': 'slowrateDos',
            'slowloris': 'slowrateDos'
        }
        
        for pattern, attack_type in attack_patterns.items():
            if pattern in filename:
                return attack_type
        
        return None
    
    def load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess all CSV files from the data directory.
        
        Returns:
            Dictionary mapping attack types to dataframes
        """
        logger.info("Loading and preprocessing data files...")
        
        # Find CSV files
        csv_files = self.find_csv_files()
        
        if not csv_files:
            logger.error("No CSV files found! Please check your data directory structure.")
            logger.error(f"Expected to find files in: {self.data_dir}")
            return {}
        
        datasets = {}
        
        # Load each file
        for attack_type, file_path in csv_files.items():
            logger.info(f"Loading {attack_type} data from {file_path}")
            
            try:
                df = self._load_and_clean_csv(file_path)
                
                if len(df) > 0:
                    datasets[attack_type] = df
                    logger.info(f"Loaded {len(df)} {attack_type} flows")
                else:
                    logger.warning(f"No data found in {file_path}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        # Update attack types list
        self.attack_types = [k for k in datasets.keys() if k != 'benign']
        logger.info(f"Found attack types: {self.attack_types}")
        
        return datasets
    
    def _load_and_clean_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load and clean a single CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Cleaned pandas DataFrame
        """
        try:
            # Read CSV with proper handling of the structure
            df = pd.read_csv(file_path, low_memory=False)
            
            logger.info(f"Original CSV shape: {df.shape}")
            logger.info(f"Original columns: {list(df.columns[:10])}...")  # Show first 10 columns
            
            # The first column is often unnamed index, we can drop it
            if df.columns[0] in ['', 'Unnamed: 0'] or df.columns[0].startswith('Unnamed'):
                df = df.drop(df.columns[0], axis=1)
                logger.info(f"Dropped unnamed index column, new shape: {df.shape}")
            
            # Clean column names (remove extra spaces, handle special characters)
            df.columns = [col.strip() for col in df.columns]
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Validate data types
            df = self._validate_and_convert_types(df)
            
            logger.info(f"Final cleaned CSV shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        
        # Fill missing numerical values with 0
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(0)
        
        # Fill missing categorical values with 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        # Exclude label columns from this processing
        categorical_cols = [col for col in categorical_cols if col not in self.label_columns]
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        return df
    
    def _validate_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        
        # Convert specific columns to appropriate types
        numeric_columns = [
            'Seq', 'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max', 'sTos', 'dTos',
            'sTtl', 'dTtl', 'sHops', 'dHops', 'TotPkts', 'SrcPkts', 'DstPkts',
            'TotBytes', 'SrcBytes', 'DstBytes', 'Offset', 'sMeanPktSz', 'dMeanPktSz',
            'Load', 'SrcLoad', 'DstLoad', 'Loss', 'SrcLoss', 'DstLoss', 'pLoss',
            'Rate', 'SrcRate', 'DstRate', 'SrcWin', 'DstWin', 'TcpRtt', 'SynAck', 'AckDat'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def extract_and_engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and engineer features from the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Processed feature matrix
        """
        logger.debug(f"Extracting features from dataframe with shape: {df.shape}")
        
        # Start with numerical features
        numerical_features = []
        categorical_features = []
        
        # Select numerical features (excluding label columns)
        for col in df.columns:
            if col not in self.label_columns and df[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
            elif col in self.categorical_feature_names and col in df.columns:
                categorical_features.append(col)
        
        # Store feature names for later use
        if not self.feature_names:
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features
            logger.info(f"Selected {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        
        # Extract numerical features
        X_numerical = df[numerical_features].values.astype(np.float32)
        
        # Handle categorical features
        X_categorical = self._encode_categorical_features(df[categorical_features])
        
        # Combine features
        if X_categorical.shape[1] > 0:
            X_combined = np.hstack([X_numerical, X_categorical])
            combined_feature_names = numerical_features + [f"{cat}_{val}" for cat in categorical_features for val in self.categorical_encoders[cat].classes_]
        else:
            X_combined = X_numerical
            combined_feature_names = numerical_features
        
        # Store feature names
        if not self.feature_names:
            self.feature_names = combined_feature_names
            logger.info(f"Total features after encoding: {len(self.feature_names)}")
        
        return X_combined
    
    def _encode_categorical_features(self, cat_df: pd.DataFrame) -> np.ndarray:
        """Encode categorical features using label encoding."""
        
        if cat_df.empty:
            return np.array([]).reshape(len(cat_df), 0)
        
        encoded_features = []
        
        for col in cat_df.columns:
            if col not in self.categorical_encoders:
                # Fit new encoder
                self.categorical_encoders[col] = LabelEncoder()
                encoded = self.categorical_encoders[col].fit_transform(cat_df[col].astype(str))
            else:
                # Use existing encoder, handle unseen categories
                try:
                    encoded = self.categorical_encoders[col].transform(cat_df[col].astype(str))
                except ValueError:
                    # Handle unseen categories by encoding them as 0
                    encoded = []
                    for val in cat_df[col].astype(str):
                        try:
                            encoded.append(self.categorical_encoders[col].transform([val])[0])
                        except ValueError:
                            encoded.append(0)  # Unknown category
                    encoded = np.array(encoded)
            
            encoded_features.append(encoded.reshape(-1, 1))
        
        if encoded_features:
            return np.hstack(encoded_features).astype(np.float32)
        else:
            return np.array([]).reshape(len(cat_df), 0)
    
    def build_indices(self, datasets: Dict[str, pd.DataFrame]):
        """
        Build FAISS indices for all attack types and benign traffic.
        
        Args:
            datasets: Dictionary of loaded datasets
        """
        logger.info("Building FAISS indices...")
        
        # First pass: collect all features to fit scalers and selectors
        all_features = []
        all_labels = []
        
        for label_idx, (dataset_name, df) in enumerate(datasets.items()):
            logger.info(f"Processing {dataset_name} ({len(df)} samples)")
            
            features = self.extract_and_engineer_features(df)
            labels = np.full(len(features), label_idx)
            
            all_features.append(features)
            all_labels.append(labels)
        
        # Combine all features
        X_all = np.vstack(all_features)
        y_all = np.hstack(all_labels)
        
        logger.info(f"Combined dataset shape: {X_all.shape}")
        
        # Fit scaler on all data
        logger.info("Fitting scaler...")
        X_scaled = self.scaler.fit_transform(X_all)
        
        # Fit feature selector
        logger.info("Selecting best features...")
        X_selected = self.feature_selector.fit_transform(X_scaled, y_all)
        
        selected_feature_indices = self.feature_selector.get_support(indices=True)
        selected_feature_names = [self.feature_names[i] for i in selected_feature_indices]
        
        logger.info(f"Selected {X_selected.shape[1]} features: {selected_feature_names[:10]}...")
        
        # Build individual indices
        current_idx = 0
        for dataset_name, df in datasets.items():
            dataset_size = len(df)
            dataset_features = X_selected[current_idx:current_idx + dataset_size]
            
            if dataset_name == 'benign':
                self._build_benign_index(dataset_features)
            else:
                self._build_malicious_index(dataset_name, dataset_features, df)
            
            current_idx += dataset_size
        
        # Store metadata
        self.index_metadata['feature_selection'] = {
            'selected_features': selected_feature_names,
            'total_features': len(self.feature_names),
            'selected_count': len(selected_feature_names)
        }
    
    def _build_benign_index(self, features: np.ndarray):
        """Build FAISS index for benign traffic."""
        dimension = features.shape[1]
        
        # Use IVF index for better performance with large datasets
        nlist = min(100, max(10, len(features) // 50))
        quantizer = faiss.IndexFlatL2(dimension)
        self.benign_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Train the index
        logger.info(f"Training benign index with {len(features)} samples...")
        self.benign_index.train(features)
        
        # Add vectors to index
        self.benign_index.add(features)
        
        self.index_metadata['benign'] = {
            'num_samples': len(features),
            'feature_dim': dimension,
            'nlist': nlist,
            'index_type': 'IVFFlat'
        }
        
        logger.info(f"Built benign index: {len(features)} samples, dimension {dimension}")
    
    def _build_malicious_index(self, attack_type: str, features: np.ndarray, original_df: pd.DataFrame):
        """Build FAISS index for a specific attack type."""
        dimension = features.shape[1]
        
        # Choose index type based on dataset size
        if len(features) < 1000:
            index = faiss.IndexFlatL2(dimension)
            index_type = 'Flat'
        else:
            nlist = min(50, max(10, len(features) // 20))
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(features)
            index_type = f'IVFFlat_nlist{nlist}'
        
        # Add vectors to index
        index.add(features)
        
        # Store index and metadata
        self.malicious_indices[attack_type] = {
            'index': index,
            'metadata': original_df.to_dict('records'),
            'features': features
        }
        
        self.index_metadata[attack_type] = {
            'num_samples': len(features),
            'feature_dim': dimension,
            'index_type': index_type
        }
        
        logger.info(f"Built {attack_type} index: {len(features)} samples, dimension {dimension}")
    
    def save_indices(self):
        """Save all FAISS indices and metadata to disk."""
        logger.info("Saving FAISS indices...")
        
        # Save benign index
        if self.benign_index:
            faiss.write_index(self.benign_index, str(self.index_dir / "benign_index.faiss"))
        
        # Save malicious indices
        for attack_type, index_data in self.malicious_indices.items():
            faiss.write_index(index_data['index'], str(self.index_dir / f"{attack_type}_index.faiss"))
            
            # Save metadata separately
            with open(self.index_dir / f"{attack_type}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'metadata': index_data['metadata'],
                    'features': index_data['features']
                }, f)
        
        # Save preprocessing components
        preprocessing_data = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'categorical_encoders': self.categorical_encoders,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'attack_types': self.attack_types,
            'index_metadata': self.index_metadata,
            'thresholds': {
                'benign_threshold': self.benign_threshold,
                'malicious_threshold': self.malicious_threshold
            }
        }
        
        with open(self.index_dir / "preprocessing.pkl", 'wb') as f:
            pickle.dump(preprocessing_data, f)
        
        # Save human-readable metadata
        with open(self.index_dir / "index_info.json", 'w') as f:
            json.dump(self.index_metadata, f, indent=2)
        
        logger.info("All indices and metadata saved successfully")
        logger.info(f"Index files saved to: {self.index_dir}")
    
    def load_indices(self):
        """Load FAISS indices and metadata from disk."""
        logger.info("Loading FAISS indices...")
        
        # Load preprocessing objects
        preprocessing_path = self.index_dir / "preprocessing.pkl"
        if not preprocessing_path.exists():
            raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")
        
        with open(preprocessing_path, 'rb') as f:
            preprocessing_data = pickle.load(f)
        
        self.scaler = preprocessing_data['scaler']
        self.feature_selector = preprocessing_data['feature_selector']
        self.categorical_encoders = preprocessing_data['categorical_encoders']
        self.feature_names = preprocessing_data['feature_names']
        self.numerical_features = preprocessing_data['numerical_features']
        self.categorical_features = preprocessing_data['categorical_features']
        self.attack_types = preprocessing_data['attack_types']
        self.index_metadata = preprocessing_data['index_metadata']
        
        if 'thresholds' in preprocessing_data:
            self.benign_threshold = preprocessing_data['thresholds']['benign_threshold']
            self.malicious_threshold = preprocessing_data['thresholds']['malicious_threshold']
        
        # Load benign index
        benign_index_path = self.index_dir / "benign_index.faiss"
        if benign_index_path.exists():
            self.benign_index = faiss.read_index(str(benign_index_path))
        
        # Load malicious indices
        self.malicious_indices = {}
        for attack_type in self.attack_types:
            index_path = self.index_dir / f"{attack_type}_index.faiss"
            metadata_path = self.index_dir / f"{attack_type}_metadata.pkl"
            
            if index_path.exists() and metadata_path.exists():
                index = faiss.read_index(str(index_path))
                
                with open(metadata_path, 'rb') as f:
                    stored_data = pickle.load(f)
                
                self.malicious_indices[attack_type] = {
                    'index': index,
                    'metadata': stored_data['metadata'],
                    'features': stored_data['features']
                }
        
        logger.info("All indices loaded successfully")
        logger.info(f"Loaded {len(self.attack_types)} attack types: {self.attack_types}")
    
    def classify_flow(self, flow_features: Dict) -> Tuple[str, float, Optional[str]]:
        """
        Classify a single network flow.
        
        Args:
            flow_features: Dictionary of flow features
            
        Returns:
            Tuple of (classification, confidence, attack_type)
        """
        try:
            # Convert features to the expected format
            feature_vector = self._dict_to_feature_vector(flow_features)
            
            # Preprocess features
            feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            feature_vector_selected = self.feature_selector.transform(feature_vector_scaled)
            
            # Check against benign index first
            if self.benign_index:
                distances, indices = self.benign_index.search(feature_vector_selected, k=5)
                avg_benign_distance = np.mean(distances[0])
                benign_similarity = 1.0 / (1.0 + avg_benign_distance)
                
                if benign_similarity > self.benign_threshold:
                    return "benign", benign_similarity, None
            
            # Check against malicious indices
            best_attack_type = None
            best_similarity = 0.0
            
            for attack_type, index_data in self.malicious_indices.items():
                distances, indices = index_data['index'].search(feature_vector_selected, k=3)
                avg_distance = np.mean(distances[0])
                similarity = 1.0 / (1.0 + avg_distance)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_attack_type = attack_type
            
            # Determine final classification
            if best_similarity > self.malicious_threshold:
                return "malicious", best_similarity, best_attack_type
            elif best_similarity > 0.4:
                return "suspicious", best_similarity, best_attack_type
            else:
                return "benign", 1.0 - best_similarity, None
                
        except Exception as e:
            logger.error(f"Error classifying flow: {e}")
            return "unknown", 0.0, None
    
    def _dict_to_feature_vector(self, flow_features: Dict) -> np.ndarray:
        """Convert flow feature dictionary to numpy array matching trained features."""
        
        # Create a mock dataframe with the same structure
        mock_data = {}
        
        # Initialize all numerical features with defaults
        for feature in self.numerical_features:
            mock_data[feature] = [flow_features.get(feature, 0.0)]
        
        # Initialize categorical features with defaults
        for feature in self.categorical_features:
            mock_data[feature] = [flow_features.get(feature, 'unknown')]
        
        mock_df = pd.DataFrame(mock_data)
        
        # Extract features using the same process
        feature_vector = self.extract_and_engineer_features(mock_df)
        
        return feature_vector[0]  # Return first (and only) row
    
    def get_similar_flows(self, flow_features: Dict, attack_type: str, k: int = 5) -> List[Dict]:
        """
        Get similar flows for a given attack type.
        
        Args:
            flow_features: Dictionary of flow features
            attack_type: Type of attack to search for
            k: Number of similar flows to return
            
        Returns:
            List of similar flow metadata
        """
        if attack_type not in self.malicious_indices:
            return []
        
        try:
            # Preprocess features
            feature_vector = self._dict_to_feature_vector(flow_features)
            feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            feature_vector_selected = self.feature_selector.transform(feature_vector_scaled)
            
            # Search for similar flows
            index_data = self.malicious_indices[attack_type]
            distances, indices = index_data['index'].search(feature_vector_selected, k=k)
            
            similar_flows = []
            for i, idx in enumerate(indices[0]):
                if idx < len(index_data['metadata']):
                    flow_data = index_data['metadata'][idx].copy()
                    flow_data['similarity'] = 1.0 / (1.0 + distances[0][i])
                    similar_flows.append(flow_data)
            
            return similar_flows
            
        except Exception as e:
            logger.error(f"Error finding similar flows: {e}")
            return []

def main():
    """Main function to build and save FAISS indices."""
    logger.info("=== Complete 5G-NIDD FAISS Indexer (Fixed for Subfolder Structure) ===")
    
    # Initialize indexer
    indexer = Complete5GNIDDFAISSIndexer()
    
    # Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data...")
    datasets = indexer.load_and_preprocess_data()
    
    if not datasets:
        logger.error("No datasets found. Please check your data directory structure.")
        logger.error("Expected structure:")
        logger.error("  data_split/")
        logger.error("    subfolder1/")
        logger.error("      benign_index.csv")
        logger.error("      attack_type_*_index.csv")
        logger.error("    subfolder2/")
        logger.error("      *.csv files")
        logger.error("  OR flat structure:")
        logger.error("  data_split/")
        logger.error("    benign_index.csv")
        logger.error("    attack_type_*_index.csv")
        return False
    
    # Print dataset summary
    logger.info("\nDataset Summary:")
    total_flows = 0
    for name, df in datasets.items():
        logger.info(f"  {name}: {len(df)} flows")
        total_flows += len(df)
    logger.info(f"  Total: {total_flows} flows")
    
    # Build indices
    logger.info("\nStep 2: Building FAISS indices...")
    start_time = time.time()
    indexer.build_indices(datasets)
    build_time = time.time() - start_time
    logger.info(f"Index building completed in {build_time:.2f} seconds")
    
    # Save indices
    logger.info("\nStep 3: Saving indices...")
    indexer.save_indices()
    
    # Test the indexer
    logger.info("\nStep 4: Testing the indexer...")
    test_indexer = Complete5GNIDDFAISSIndexer()
    test_indexer.load_indices()
    
    # Create sample flows for testing based on actual data structure
    test_flows = [
        {
            # Benign-like flow
            'Seq': 1000, 'Dur': 0.1, 'Mean': 0.05, 'Proto': 'tcp', 'TotPkts': 10,
            'SrcBytes': 500, 'TotBytes': 1000, 'sMeanPktSz': 50, 'sHops': 5,
            'TcpRtt': 10.0, 'AckDat': 100, 'State': 'CON'
        },
        {
            # Attack-like flow (ICMP flood pattern)
            'Seq': 2000, 'Dur': 0.001, 'Mean': 0.001, 'Proto': 'icmp', 'TotPkts': 2,
            'SrcBytes': 42, 'TotBytes': 84, 'sMeanPktSz': 42, 'sHops': 1,
            'TcpRtt': 0.0, 'AckDat': 0, 'State': 'ECO'
        },
        {
            # Scan-like flow
            'Seq': 3000, 'Dur': 0.01, 'Mean': 0.005, 'Proto': 'tcp', 'TotPkts': 1,
            'SrcBytes': 64, 'TotBytes': 64, 'sMeanPktSz': 64, 'sHops': 10,
            'TcpRtt': 0.1, 'AckDat': 0, 'State': 'REQ'
        }
    ]
    
    logger.info("\nTest Results:")
    for i, flow in enumerate(test_flows):
        classification, confidence, attack_type = test_indexer.classify_flow(flow)
        logger.info(f"  Test flow {i+1}: {classification} (confidence: {confidence:.3f}, attack_type: {attack_type})")
    
    # Print final statistics
    logger.info(f"\nIndexing Summary:")
    logger.info(f"  Total build time: {build_time:.2f} seconds")
    logger.info(f"  Indices saved to: {indexer.index_dir}")
    logger.info(f"  Attack types indexed: {len(indexer.attack_types)}")
    
    if indexer.benign_index:
        logger.info(f"  Benign flows indexed: {indexer.index_metadata['benign']['num_samples']}")
    
    for attack_type in indexer.attack_types:
        if attack_type in indexer.index_metadata:
            count = indexer.index_metadata[attack_type]['num_samples']
            logger.info(f"  {attack_type} flows indexed: {count}")
    
    logger.info("\n=== FAISS indexing completed successfully! ===")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)