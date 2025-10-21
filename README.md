----- BEGIN README.md -----
# TRICOTTET-GAM-CG2

TRICOTTET-GAM-CG2 is an astrophysical data analysis project written in Python. It includes tools for statistical testing, spherical trigonometry, and specialized physics computations. The project also automatically generates reports and documentation.

## Project Structure

```
TRICOTTET-GAM-CG2/
├── notebooks/          # Jupyter notebooks for interactive analysis and visualization
├── output/             # Generated outputs (e.g., LaTeX reports, PDFs, data files)
├── src/                # Source code for analysis and utilities
│   ├── __init__.py     # Package initializer
│   ├── main.py         # Main entry point for running the analysis
│   ├── astro_utils.py  # Core astrophysical computation functions
│   └── utils/          # Utility modules
│       ├── __init__.py
│       ├── spherical_utils.py
│       ├── stats_utils.py
│       └── physics_utils.py
├── requirements.txt    # List of Python dependencies
└── README.md           # This file
```

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/MattACDC/TRICOTTET-GAM-CG2.git
   ```

2. **Navigate to the project directory:**
   ```
   cd TRICOTTET-GAM-CG2
   ```

3. **(Optional) Create and activate a virtual environment:**
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

- **Run the Analysis:**  
  Execute the main script to run all analyses:
  ```
  python -m src.main
  ```

- **Generate Reports:**  
  The main analysis calls the report generation code that creates a LaTeX report in the `output/` directory.  
  To run the report generation separately:
  ```
  python -m src.generate_report
  ```

- **Interactive Exploration:**  
  Open notebooks in the `notebooks/` directory with Jupyter Notebook or VS Code's Jupyter extension for interactive data exploration.

## Documentation

Documentation is automatically generated from docstrings using tools like [pdoc](https://pdoc.dev/) or [Sphinx](https://www.sphinx-doc.org/). For example, to generate HTML documentation with pdoc, run:

```
pdoc --html src --output-dir docs --force
```

Then open `docs/index.html` in your browser.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [your.email@example.com](mailto:your.email@example.com).
----- END README.md -----