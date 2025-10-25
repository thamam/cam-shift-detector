# cam-shift-detector Project

## Sample Images for Testing

### Overview
A diverse sample of 50 real images from DAF (Digital Agriculture Framework) sites has been added to the project for testing camera shift detection features. These images were systematically sampled from the greenpipe datasets to ensure representative coverage.

### Dataset Distribution

**Total Images**: 50 (systematically sampled from 30,205 available images)

#### 1. OF_JERUSALEM Dataset
- **Location**: `sample_images/of_jerusalem/`
- **Count**: 23 images
- **Source**: 9bc4603f-0d21-4f60-afea-b6343d372034 (14,039 total images)
- **Sampling**: Every 610th image (systematic sampling)

#### 2. CARMIT Dataset
- **Location**: `sample_images/carmit/`
- **Count**: 17 images
- **Source**: e2336087-0143-4895-bc12-59b8b8f97790 (10,430 total images)
- **Sampling**: Every 613th image (systematic sampling)

#### 3. GAD Dataset
- **Location**: `sample_images/gad/`
- **Count**: 10 images
- **Source**: f10f17d4-ac26-4c28-b601-06c64b8a22a4 (5,736 total images)
- **Sampling**: Every 574th image (systematic sampling)

### Purpose
These sample images provide real-world data for:
- Testing camera shift detection algorithms
- Validating feature extraction and matching
- Benchmarking performance on actual DAF site imagery
- Ensuring algorithms work on diverse agricultural scenes

### Source Data
All images sourced from: `/home/thh3/data/greenpipe/`

Dataset mappings defined in: `/home/thh3/data/greenpipe/datasets_uuid_legend.md`
