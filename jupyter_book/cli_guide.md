# Command-Line Interface Guide

Row2Vec provides a comprehensive CLI for batch processing and production workflows.

## CLI Commands Overview

Row2Vec offers three main commands:

1. **`row2vec`** - Direct embedding generation (most common)
2. **`row2vec-train`** - Train and save models
3. **`row2vec-embed`** - Apply saved models to new data

## Installation

The CLI is included with Row2Vec:

```bash
pip install row2vec[cli]  # Include CLI dependencies
```

Or install CLI dependencies manually:
```bash
pip install click rich pyyaml
```

## Basic Usage Examples

### Quick Embeddings

```bash
# Simple 2D embeddings
row2vec data.csv --output embeddings.csv

# With specific dimensions
row2vec data.csv --dimensions 10 --output embeddings.csv

# Target-based embeddings
row2vec data.csv --target-column category --output embeddings.csv
```

### Working with Different Modes

```bash
# Neural network (default)
row2vec data.csv --mode unsupervised --dimensions 5 --output neural_emb.csv

# PCA for fast linear reduction
row2vec data.csv --mode pca --dimensions 5 --output pca_emb.csv

# t-SNE for visualization
row2vec data.csv --mode tsne --dimensions 2 --output tsne_emb.csv

# UMAP for general purpose
row2vec data.csv --mode umap --dimensions 3 --output umap_emb.csv
```

## Advanced Configuration

### Using Configuration Files

Create a YAML configuration file for complex setups:

```yaml
# config.yaml
neural:
  max_epochs: 100
  batch_size: 128
  dropout_rate: 0.3
  hidden_units: [512, 256]
  early_stopping: true

preprocessing:
  categorical_encoding_strategy: "adaptive"
  numeric_scaling: "standard"
  handle_missing: "median"

scaling:
  method: "minmax"
  feature_range: [-1.0, 1.0]

logging:
  level: "INFO"
  enable_performance: true
```

Use the configuration:
```bash
row2vec data.csv --config config.yaml --output embeddings.csv
```

### Large Dataset Processing

```bash
# Sample large datasets
row2vec huge_data.csv --sample-size 10000 --output embeddings.csv

# Batch processing with specific settings
row2vec big_data.parquet \
  --dimensions 20 \
  --batch-size 512 \
  --sample-size 50000 \
  --output embeddings.parquet
```

## Training and Model Management

### Train Models

```bash
# Basic model training
row2vec-train data.csv --output model.pkl

# Advanced training
row2vec-train data.csv \
  --dimensions 15 \
  --mode unsupervised \
  --max-epochs 100 \
  --batch-size 256 \
  --validation-split 0.2 \
  --save-embeddings \
  --output production_model.pkl
```

### Apply Trained Models

```bash
# Basic inference
row2vec-embed new_data.csv --model model.py --output embeddings.csv

# With validation
row2vec-embed data.csv \
  --model trained_model.py \
  --strict-validation \
  --output embeddings.parquet
```

## File Format Support

Row2Vec CLI supports multiple formats:

**Input formats:**
- CSV (`.csv`)
- Parquet (`.parquet`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- TSV (`.tsv`)

**Output formats:**
- CSV (`.csv`) - Default, widely compatible
- Parquet (`.parquet`) - Recommended for large datasets
- Excel (`.xlsx`) - For reports and analysis
- JSON (`.json`) - For APIs and web services

```bash
# Different format examples
row2vec data.parquet --output embeddings.parquet  # Parquet to Parquet
row2vec data.xlsx --output embeddings.csv         # Excel to CSV
row2vec data.json --output embeddings.json        # JSON to JSON
```

## Real-World Workflow Examples

### Data Science Pipeline

```bash
# 1. Explore with quick embeddings
row2vec exploration_data.csv --dimensions 2 --mode tsne --output explore.csv

# 2. Train production model
row2vec-train clean_data.csv \
  --dimensions 50 \
  --max-epochs 200 \
  --validation-split 0.3 \
  --save-embeddings \
  --output production_model.pkl

# 3. Apply to new data
row2vec-embed daily_data.csv \
  --model production_model.py \
  --validate-schema \
  --output daily_embeddings.csv
```

### Customer Analytics

```bash
# Customer segmentation embeddings
row2vec customer_data.csv \
  --dimensions 10 \
  --mode unsupervised \
  --categorical-strategy adaptive \
  --numeric-scaling robust \
  --output customer_segments.csv

# Category analysis
row2vec customer_data.csv \
  --mode target \
  --target-column customer_type \
  --dimensions 3 \
  --output customer_types.csv
```

### A/B Testing Setup

```bash
# Train baseline model
row2vec-train historical_data.csv \
  --config baseline_config.yaml \
  --output baseline_model.pkl

# Generate embeddings for test groups
row2vec-embed test_group_a.csv \
  --model baseline_model.py \
  --output test_a_embeddings.csv

row2vec-embed test_group_b.csv \
  --model baseline_model.py \
  --output test_b_embeddings.csv
```

## Monitoring and Debugging

### Verbose Output

```bash
# Enable detailed logging
row2vec data.csv --verbose --output embeddings.csv

# Custom log levels and files
row2vec data.csv \
  --log-level DEBUG \
  --log-file training.log \
  --output embeddings.csv
```

### Performance Monitoring

```bash
# Time execution
time row2vec large_data.csv --output embeddings.csv

# Monitor memory usage
row2vec data.csv --log-level INFO --output embeddings.csv 2>&1 | grep memory
```

### Error Handling

```bash
# Relaxed validation for messy data
row2vec-embed messy_data.csv \
  --model model.py \
  --no-strict-validation \
  --output embeddings.csv

# Skip problematic rows (if implemented)
row2vec problematic_data.csv \
  --handle-errors skip \
  --output embeddings.csv
```

## Integration Examples

### Shell Scripting

```bash
#!/bin/bash
# process_daily_data.sh

DATA_DIR="/data/daily"
MODEL_PATH="/models/production_model.py"
OUTPUT_DIR="/embeddings/daily"

for file in $DATA_DIR/*.csv; do
  basename=$(basename "$file" .csv)
  echo "Processing $basename..."

  row2vec-embed "$file" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR/${basename}_embeddings.csv" \
    --log-file "$OUTPUT_DIR/${basename}.log"
done

echo "Daily processing complete!"
```

### Python Integration

```bash
# Generate embeddings then process with Python
row2vec data.csv --output embeddings.csv
python analyze_embeddings.py embeddings.csv
```

### Docker Usage

```dockerfile
# Dockerfile
FROM python:3.10

RUN pip install row2vec

COPY . /app
WORKDIR /app

# Process data in container
CMD ["row2vec", "input/data.csv", "--output", "output/embeddings.csv"]
```

```bash
# Build and run
docker build -t row2vec-processor .
docker run -v $(pwd)/data:/app/input -v $(pwd)/output:/app/output row2vec-processor
```

## Performance Tips

### Optimization Strategies

```bash
# Large datasets: use sampling
row2vec huge_data.csv \
  --sample-size 100000 \
  --batch-size 1024 \
  --output embeddings.csv

# Fast iteration: use PCA
row2vec data.csv --mode pca --dimensions 10 --output quick_emb.csv

# Production quality: more epochs
row2vec-train data.csv \
  --max-epochs 500 \
  --early-stopping \
  --output final_model.pkl
```

### Memory Management

```bash
# Process in chunks for very large files
split -l 10000 huge_data.csv chunk_
for chunk in chunk_*; do
  row2vec "$chunk" --output "${chunk}_emb.csv"
done
cat chunk_*_emb.csv > all_embeddings.csv
```

## Troubleshooting

### Common Issues

**Memory errors:**
```bash
# Reduce batch size and sample data
row2vec large_data.csv \
  --sample-size 10000 \
  --batch-size 32 \
  --output embeddings.csv
```

**Slow performance:**
```bash
# Use PCA for quick results
row2vec data.csv --mode pca --dimensions 5 --output fast_emb.csv

# Or reduce epochs
row2vec data.csv --max-epochs 10 --output quick_emb.csv
```

**Schema validation errors:**
```bash
# Disable strict validation
row2vec-embed data.csv \
  --model model.py \
  --no-strict-validation \
  --output embeddings.csv
```

### Getting Help

```bash
# General help
row2vec --help

# Command-specific help
row2vec-train --help
row2vec-embed --help

# Version information
row2vec --version
```

## Configuration Reference

Complete YAML configuration example:

```yaml
# Complete configuration example
neural:
  max_epochs: 100
  batch_size: 128
  dropout_rate: 0.25
  hidden_units: [512, 256, 128]
  early_stopping: true
  early_stopping_patience: 10

classical:
  n_neighbors: 20        # UMAP
  perplexity: 50.0      # t-SNE
  min_dist: 0.05        # UMAP
  n_iter: 1000          # t-SNE

preprocessing:
  categorical_encoding_strategy: "adaptive"
  numeric_scaling: "standard"
  handle_missing: "adaptive"

scaling:
  method: "minmax"
  feature_range: [-1.0, 1.0]

logging:
  level: "INFO"
  enable_performance: true
  enable_memory: false
```

The CLI provides powerful batch processing capabilities while maintaining Row2Vec's simplicity. Use it for production workflows, automation, and large-scale data processing.

## Advanced Workflows

### Complete ML Pipeline

```bash
# 1. Explore with quick embeddings
row2vec exploration_data.csv --dimensions 10 --output exploration.csv

# 2. Train production model
row2vec-train training_data.csv \
  --target outcome \
  --validation-split 0.2 \
  --max-epochs 100 \
  --output production_model.pkl

# 3. Apply to new data
row2vec-embed new_data.csv \
  --model production_model.py \
  --output predictions.csv
```

### Production Deployment

```bash
# 1. Model training with validation
row2vec-train historical_data.csv \
  --target conversion \
  --validation-split 0.3 \
  --max-epochs 150 \
  --batch-size 256 \
  --categorical-strategy adaptive \
  --output production_model.pkl

# 2. Daily batch inference
row2vec-embed daily_data.csv \
  --model production_model.py \
  --validate-schema \
  --output daily_embeddings.parquet

# 3. Real-time inference
row2vec-embed realtime_batch.csv \
  --model production_model.py \
  --no-strict-validation \
  --output realtime_embeddings.json
```

## Schema Validation

The CLI provides robust schema validation for production use:

```bash
# Strict validation (default)
row2vec-embed data.csv --model model.py --strict-validation --output embeddings.csv

# Relaxed validation for data drift
row2vec-embed data.csv --model model.py --no-strict-validation --output embeddings.csv
```

## Performance Tips

1. **Use Parquet for large datasets**: Better compression and faster I/O
2. **Sample large datasets**: Use `--sample-size` to manage memory
3. **Enable early stopping**: Reduces training time while maintaining quality
4. **Use appropriate batch sizes**: Larger batches for GPUs, smaller for CPUs
5. **Cache trained models**: Reuse models across similar datasets
6. **Monitor memory usage**: Enable performance logging for optimization

## Troubleshooting

### Common Issues

**Import errors**: Ensure all dependencies are installed
```bash
pip install click rich pyyaml pandas scikit-learn tensorflow
```

**Memory errors**: Use sampling for large datasets
```bash
row2vec large_data.csv --sample-size 10000 --output embeddings.csv
```

**Schema validation failures**: Use relaxed validation for data drift
```bash
row2vec-embed data.csv --model model.py --no-strict-validation --output embeddings.csv
```

**Training convergence**: Adjust epochs and early stopping
```bash
row2vec-train data.csv --max-epochs 200 --no-early-stopping --output model.pkl
```

### Debug Mode

Enable verbose output and logging for troubleshooting:

```bash
row2vec data.csv --verbose --log-level DEBUG --log-file debug.log --output embeddings.csv
```

## Next Steps

- 📖 [API Reference](api_reference.md) - Complete function documentation
- 🏠 [Examples](titanic_example.md) - Return to interactive examples
- ⚙️ [Advanced Features](advanced_features.md) - Neural architecture search and more
