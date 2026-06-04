.PHONY: data install lint clean

# Regenerate synthetic data
data:
	python src/generate_data.py --output-dir data/raw

# Install dependencies
install:
	pip install -r requirements.txt

# Run notebook 1 headlessly (requires nbconvert)
run-hito1:
	jupyter nbconvert --to notebook --execute notebooks/01_eda_survival.ipynb \
	    --output notebooks/01_eda_survival_executed.ipynb

# Run notebook 2 headlessly
run-hito2:
	jupyter nbconvert --to notebook --execute notebooks/02_ltv_models.ipynb \
	    --output notebooks/02_ltv_models_executed.ipynb

# Run notebook 3 headlessly
run-hito3:
	jupyter nbconvert --to notebook --execute notebooks/03_pricing_bridge.ipynb \
	    --output notebooks/03_pricing_bridge_executed.ipynb

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -f notebooks/*_executed.ipynb
