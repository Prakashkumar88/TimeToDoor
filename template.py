import os
from pathlib import Path

while True:
    project_name = input("Enter the project name (or type 'exit' to quit): ").strip()
    if project_name.lower() == 'exit':
        print("Exiting the script.")
        break
    if not project_name:
        print("Project name cannot be empty. Please try again.")
        continue

    files = [
        f"{project_name}/__init__.py",
        f"{project_name}/components/__init__.py",
        f"{project_name}/config/__init__.py",
        f"{project_name}/constants/__init__.py",
        f"{project_name}/entity/__init__.py",
        f"{project_name}/exceptions/__init__.py",
        f"{project_name}/logger/__init__.py",
        f"{project_name}/pipelines/__init__.py",
        f"{project_name}/utils/__init__.py",
        "config/config.yaml",
        "schema.yaml",
        "app.py",
        "main.py",
        "logs.py",
        "exceptions.py",
        "setup.py",
    ]

    for file_path in files:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# Auto-generated file.\n")
            print(f"Created: {file_path}")
        else:
            print(f"Already exists: {file_path}")

    print("Project structure created successfully.")
    break
