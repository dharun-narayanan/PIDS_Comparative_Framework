#!/usr/bin/env python3
"""
Verification script for PIDS Comparative Framework
Tests that all models can be imported successfully
"""

def test_imports():
    """Test all model imports"""
    print("=" * 60)
    print("PIDS Framework - Model Import Verification")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test MAGIC
    print("Testing MAGIC imports...")
    try:
        from models.implementations.magic import (
            build_model, GMAEModel, GAT,
            sce_loss, masked_mse_loss,
            evaluate_entity_level_using_knn
        )
        results['magic'] = '✅ SUCCESS'
        print("  ✅ MAGIC imports successful")
    except Exception as e:
        results['magic'] = f'❌ FAILED: {e}'
        print(f"  ❌ MAGIC imports failed: {e}")
    print()
    
    # Test Kairos
    print("Testing Kairos imports...")
    try:
        from models.implementations.kairos import (
            KairosModel, GraphAttentionEmbedding, LinkPredictor,
            TimeEncoder, setup_kairos_model
        )
        results['kairos'] = '✅ SUCCESS'
        print("  ✅ Kairos imports successful")
    except Exception as e:
        results['kairos'] = f'❌ FAILED: {e}'
        print(f"  ❌ Kairos imports failed: {e}")
    print()
    
    # Test Orthrus
    print("Testing Orthrus imports...")
    try:
        from models.implementations.orthrus import (
            Orthrus, OrthrusEncoder,
            get_encoder, get_decoders,
            setup_orthrus_model
        )
        results['orthrus'] = '✅ SUCCESS'
        print("  ✅ Orthrus imports successful")
    except Exception as e:
        results['orthrus'] = f'❌ FAILED: {e}'
        print(f"  ❌ Orthrus imports failed: {e}")
    print()
    
    # Test Continuum_FL
    print("Testing Continuum_FL imports...")
    try:
        from models.implementations.continuum_fl import (
            STGNN_AutoEncoder, STGNN, GAT, RNN_Cells,
            sce_loss, setup_continuum_fl_model
        )
        results['continuum_fl'] = '✅ SUCCESS'
        print("  ✅ Continuum_FL imports successful")
    except Exception as e:
        results['continuum_fl'] = f'❌ FAILED: {e}'
        print(f"  ❌ Continuum_FL imports failed: {e}")
    print()
    
    # Test ThreaTrace
    print("Testing ThreaTrace imports...")
    try:
        from models.implementations.threatrace import (
            ThreaTraceModel, SketchGenerator,
            setup_threatrace_model
        )
        results['threatrace'] = '✅ SUCCESS'
        print("  ✅ ThreaTrace imports successful")
    except Exception as e:
        results['threatrace'] = f'❌ FAILED: {e}'
        print(f"  ❌ ThreaTrace imports failed: {e}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for v in results.values() if '✅' in v)
    total_count = len(results)
    
    for model, status in results.items():
        print(f"{model:15s}: {status}")
    
    print()
    print(f"Total: {success_count}/{total_count} models successfully imported")
    print("=" * 60)
    
    return success_count == total_count


def test_model_registry():
    """Test the model registry"""
    print()
    print("=" * 60)
    print("Model Registry Check")
    print("=" * 60)
    
    try:
        from models.base_model import ModelRegistry
        registered_models = ModelRegistry.list_models()
        print(f"Registered models: {registered_models}")
        print(f"Total models: {len(registered_models)}")
        
        expected = ['magic', 'kairos', 'orthrus', 'continuum_fl', 'threatrace']
        missing = [m for m in expected if m not in registered_models]
        
        if missing:
            print(f"⚠️  Missing models: {missing}")
        else:
            print("✅ All expected models registered!")
        
        return len(missing) == 0
        
    except Exception as e:
        print(f"❌ Model registry check failed: {e}")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("PIDS Comparative Framework - Verification Suite")
    print("=" * 60)
    print()
    
    import_success = test_imports()
    registry_success = test_model_registry()
    
    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if import_success and registry_success:
        print("🎉 ALL TESTS PASSED! Framework is ready to use! 🎉")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
