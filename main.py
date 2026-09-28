from __future__ import annotations
import argparse
from pathlib    import Path
import tkinter  as tk
from tkinter    import filedialog, messagebox
#
# Load custom function
from csv_workflow import CSVWorkflow
from segmenter_app import ImageSegmenterApp
#
def choose_csv() -> str | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select CSV containing File_path and Segment",
                                      filetypes=[("CSV or text files", "*.csv *.txt"), ("All files", "*.*")])
    root.destroy()
    return path or None
#
def resolve_csv_argument(args: argparse.Namespace) -> str | None:
    if args.csv:
        return str(Path(args.csv).expanduser())
    if args.csv_file:
        csv_file = Path(args.csv_file).expanduser()
        if csv_file.is_absolute():
            return str(csv_file)
        return str(Path(args.path_to_csv).expanduser() / csv_file)
    return choose_csv()
#
def main():
    parser = argparse.ArgumentParser(description="Simple2DSeg  — Python Image Segmenter")
    parser.add_argument("csv", nargs="?", help="Full CSV path. If omitted, a file picker opens unless --csv_file is supplied.")
    parser.add_argument("--path_to_csv", "--path-to-csv", default=".", help="Directory containing the CSV file.")
    parser.add_argument("--csv_file", "--csv-file", default=None, help="CSV filename inside --path_to_csv.")
    args = parser.parse_args()
    #
    csv_path = resolve_csv_argument(args)
    if not csv_path:
        print("No CSV file selected. Program terminated.")
        return 0
    #
    try:
        workflow = CSVWorkflow(csv_path)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("CSV error", str(e))
        root.destroy()
        return 2
    #
    print(f"CSV loaded: {workflow.csv_path}")
    print(f"Number of cases: {len(workflow.df)}")
    print("Mask_path column detected." if workflow.has_mask_path else "Mask_path column not detected.")
    app = ImageSegmenterApp(workflow)
    app.mainloop()
    return 0
#
if __name__ == "__main__":
    raise SystemExit(main())
