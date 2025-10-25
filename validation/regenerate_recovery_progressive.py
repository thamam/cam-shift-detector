#!/usr/bin/env python3
"""
Regenerate recovery and progressive patterns with corrected annotations
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent))

from generate_stage2_data import (
    generate_pattern_3_progressive,
    generate_pattern_5_recovery,
    select_baseline_images
)
from stage2_test_harness import Stage2TestHarness

def main():
    print("=" * 80)
    print("Regenerating Recovery & Progressive Patterns")
    print("=" * 80)

    # Paths
    output_dir = Path("validation/stage2_data")
    sample_images_dir = Path("sample_images")

    # Initialize harness (threshold from config.json)
    harness = Stage2TestHarness(threshold_px=1.5)

    # Load baseline images
    print("\n📷 Selecting baseline images...")
    baseline_images = select_baseline_images(sample_images_dir, count=10)
    print(f"   Selected {len(baseline_images)} baseline images")

    # Backup old patterns
    print("\n💾 Backing up old patterns...")
    patterns_to_regenerate = ['pattern_3_progressive', 'pattern_5_recovery']

    for pattern in patterns_to_regenerate:
        pattern_dir = output_dir / pattern
        backup_dir = output_dir / f"{pattern}_backup"

        if pattern_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(str(pattern_dir), str(backup_dir))
            print(f"   Backed up {pattern} → {pattern}_backup")

    # Regenerate progressive pattern
    print("\n" + "=" * 80)
    print("Regenerating Progressive Pattern (0.5px Incremental Steps)")
    print("=" * 80)
    generate_pattern_3_progressive(harness, baseline_images, output_dir)

    # Regenerate recovery pattern
    print("\n" + "=" * 80)
    print("Regenerating Recovery Pattern (5px→0 Gradual Return)")
    print("=" * 80)
    generate_pattern_5_recovery(harness, baseline_images, output_dir)

    print("\n" + "=" * 80)
    print("✅ Regeneration Complete!")
    print("=" * 80)
    print(f"\n📁 Regenerated sequences in: {output_dir}")
    print(f"📁 Old backups in: {output_dir}/*_backup/")
    print("\n🔄 Ready to re-run Stage 2 validation")

if __name__ == "__main__":
    main()
