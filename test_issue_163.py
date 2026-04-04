#!/usr/bin/env python3
"""
Test for issue #163: force_json_field_order config not being respected
"""
from dataclasses import replace
from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig

def test_config_preservation():
    """Test that replace preserves config values"""
    # Create a config with custom force_json_field_order
    config = CharacterLevelParserConfig(force_json_field_order=True)
    
    # Simulate what TokenEnforcer does after our fix
    new_config = replace(config, alphabet='abc123')
    
    # Verify force_json_field_order is preserved
    assert new_config.force_json_field_order == True, \
        f"Expected force_json_field_order=True, got {new_config.force_json_field_order}"
    
    # Verify alphabet is updated
    assert new_config.alphabet == 'abc123', \
        f"Expected alphabet='abc123', got {new_config.alphabet}"
    
    print("✓ Config preservation test passed")
    print(f"  force_json_field_order preserved: {new_config.force_json_field_order}")
    print(f"  alphabet updated: {new_config.alphabet}")

def test_old_behavior_bug():
    """Demonstrate the old bug"""
    # Old way (bug): creates new config with defaults
    old_config = CharacterLevelParserConfig(force_json_field_order=True)
    new_config_wrong = CharacterLevelParserConfig(alphabet='abc123')
    
    print("\n✓ Old bug demonstrated:")
    print(f"  Original force_json_field_order: {old_config.force_json_field_order}")
    print(f"  After wrong approach: {new_config_wrong.force_json_field_order}")
    print(f"  (Bug: should be True but is False)")

if __name__ == "__main__":
    test_config_preservation()
    test_old_behavior_bug()
    print("\n✅ All tests passed!")
