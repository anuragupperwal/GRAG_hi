
venv: python3 -m venv graphrag_env


activate: source graphrag_env/bin/activate; 
        conda deactivate (if some other env or base env is activated)