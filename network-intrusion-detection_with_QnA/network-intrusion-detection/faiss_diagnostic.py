#!/usr/bin/env python3
"""
Diagnostic script to check FAISS indices and data integrity.
Run this to identify why classifications are all benign.
"""

import os
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_data_files():
    """Check the CSV data files for issues."""
    logger.info("=== Diagnosing CSV Data Files ===")
    
    data_dir = Path("data_split")
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist!")
        return False
    
    csv_files = list(data_dir.rglob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files:")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"  {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
            
            if len(df) == 0:
                logger.error(f"    ❌ EMPTY FILE: {csv_file}")
            elif len(df) < 100:
                logger.warning(f"    ⚠️  Very small file: {csv_file}")
            else:
                logger.info(f"    ✓ OK: {csv_file}")
                
            # Check for required columns
            required_cols = ['Proto', 'TotPkts', 'TotBytes', 'Dur']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"    Missing columns: {missing_cols}")
                
        except Exception as e:
            logger.error(f"    ❌ Error reading {csv_file}: {e}")
    
    return len(csv_files) > 0

def diagnose_faiss_indices():
    """Check FAISS indices for integrity."""
    logger.info("\n=== Diagnosing FAISS Indices ===")
    
    index_dir = Path("faiss_indices")
    if not index_dir.exists():
        logger.error(f"Index directory {index_dir} does not exist!")
        return False
    
    # Check preprocessing file
    preprocessing_file = index_dir / "preprocessing.pkl"
    if not preprocessing_file.exists():
        logger.error("❌ preprocessing.pkl not found - indices not built!")
        return False
    
    try:
        import pickle
        with open(preprocessing_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"✓ Preprocessing data loaded")
        logger.info(f"  Attack types: {data.get('attack_types', [])}")
        logger.info(f"  Feature names: {len(data.get('feature_names', []))} features")
        logger.info(f"  Numerical features: {len(data.get('numerical_features', []))}")
        logger.info(f"  Categorical features: {len(data.get('categorical_features', []))}")
        
    except Exception as e:
        logger.error(f"❌ Error loading preprocessing data: {e}")
        return False
    
    # Check individual FAISS index files
    benign_index = index_dir / "benign_index.faiss"
    if benign_index.exists():
        try:
            index = faiss.read_index(str(benign_index))
            logger.info(f"✓ Benign index: {index.ntotal} vectors, dimension {index.d}")
            if index.ntotal == 0:
                logger.error("❌ Benign index is EMPTY!")
        except Exception as e:
            logger.error(f"❌ Error reading benign index: {e}")
    else:
        logger.error("❌ Benign index file not found!")
    
    # Check attack indices
    attack_types = data.get('attack_types', [])
    for attack_type in attack_types:
        attack_index_file = index_dir / f"{attack_type}_index.faiss"
        if attack_index_file.exists():
            try:
                index = faiss.read_index(str(attack_index_file))
                logger.info(f"✓ {attack_type} index: {index.ntotal} vectors, dimension {index.d}")
                if index.ntotal == 0:
                    logger.error(f"❌ {attack_type} index is EMPTY!")
            except Exception as e:
                logger.error(f"❌ Error reading {attack_type} index: {e}")
        else:
            logger.error(f"❌ {attack_type} index file not found!")
    
    return True

def test_feature_preprocessing():
    """Test feature preprocessing pipeline."""
    logger.info("\n=== Testing Feature Preprocessing ===")
    
    try:
        from complete_faiss_indexer import Complete5GNIDDFAISSIndexer
        
        indexer = Complete5GNIDDFAISSIndexer()
        indexer.load_indices()
        
        # Create a test flow
        test_flow = {
            'Seq': 1000, 'Dur': 0.1, 'Proto': 'tcp', 'TotPkts': 10,
            'SrcBytes': 500, 'TotBytes': 1000, 'State': 'CON',
            'sTos': 0, 'dTos': 0, 'sDSb': 'cs0', 'dDSb': 'cs0',
            'sTtl': 64, 'dTtl': 64, 'sHops': 10, 'dHops': 10,
            'Cause': 'Start', 'SrcPkts': 5, 'DstPkts': 5,
            'DstBytes': 500, 'Offset': 1000, 'sMeanPktSz': 100,
            'dMeanPktSz': 100, 'Load': 0, 'SrcLoad': 0, 'DstLoad': 0,
            'Loss': 0, 'SrcLoss': 0, 'DstLoss': 0, 'pLoss': 0.0,
            'SrcGap': 0, 'DstGap': 0, 'Rate': 10000, 'SrcRate': 5000,
            'DstRate': 5000, 'SrcWin': 0, 'DstWin': 0, 'sVid': 0,
            'dVid': 0, 'SrcTCPBase': 0, 'DstTCPBase': 0, 'TcpRtt': 0.1,
            'SynAck': 0, 'AckDat': 1000
        }
        
        logger.info("✓ Indexer loaded successfully")
        
        # Test feature vector creation
        feature_vector = indexer._create_feature_vector_from_dict(test_flow)
        logger.info(f"✓ Feature vector created: shape {feature_vector.shape}")
        
        if np.all(feature_vector == 0):
            logger.error("❌ Feature vector is all zeros!")
        else:
            logger.info(f"  Feature vector stats: min={feature_vector.min():.3f}, max={feature_vector.max():.3f}")
        
        # Test preprocessing pipeline
        vec_scaled = indexer.scaler.transform(feature_vector.reshape(1, -1))
        vec_selected = indexer.feature_selector.transform(vec_scaled)
        logger.info(f"✓ Preprocessing works: scaled shape {vec_scaled.shape}, selected shape {vec_selected.shape}")
        
        # Test similarity calculation
        benign_score = indexer._calculate_similarity_score(vec_selected, 'benign')
        logger.info(f"✓ Benign similarity score: {benign_score}")
        
        if benign_score == 0.0:
            logger.error("❌ Benign similarity score is 0.0 - this is the problem!")
            
            # Debug the FAISS search
            if indexer.benign_index is not None:
                k = min(5, indexer.benign_index.ntotal)
                distances, indices = indexer.benign_index.search(vec_selected, k=k)
                logger.info(f"  FAISS search returned distances: {distances[0]}")
                logger.info(f"  FAISS search returned indices: {indices[0]}")
                
                if len(distances[0]) > 0:
                    avg_distance = np.mean(distances[0])
                    logger.info(f"  Average distance: {avg_distance}")
                    logger.info(f"  Exponential similarity: {np.exp(-avg_distance / 10.0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in feature preprocessing test: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fixes():
    """Suggest fixes based on diagnosis."""
    logger.info("\n=== Suggested Fixes ===")
    
    data_ok = diagnose_data_files()
    indices_ok = diagnose_faiss_indices()
    
    if not data_ok:
        logger.info("1. ❌ DATA ISSUE: Check your data_split directory")
        logger.info("   - Ensure CSV files are present and not empty")
        logger.info("   - Files should be named like: attack_type_*_index.csv")
        logger.info("   - Run: python fixed_setup_script.py --inspect")
        
    if not indices_ok:
        logger.info("2. ❌ INDEX ISSUE: Rebuild FAISS indices")
        logger.info("   - Run: python complete_faiss_indexer.py")
        logger.info("   - Or: python fixed_setup_script.py --setup")
    
    preprocessing_ok = test_feature_preprocessing()
    if not preprocessing_ok:
        logger.info("3. ❌ PREPROCESSING ISSUE: Feature pipeline broken")
        logger.info("   - Check complete_faiss_indexer.py for errors")
        logger.info("   - Ensure data format matches expected schema")

def main():
    """Run full diagnostic."""
    logger.info("🔍 Running FAISS Index Diagnostic...")
    
    suggest_fixes()
    
    logger.info("\n=== Next Steps ===")
    logger.info("1. Fix any data issues identified above")
    logger.info("2. Rebuild FAISS indices if needed")
    logger.info("3. Test classification again")
    logger.info("4. If still failing, check the 5G-NIDD dataset paper for proper data format")

if __name__ == "__main__":
    main()