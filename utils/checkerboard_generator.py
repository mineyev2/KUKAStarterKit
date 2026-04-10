import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages


def generate_checkerboard_pdf(
    square_size_mm, rows=6, cols=9, output_filename="checkerboard.pdf"
):
    """
    Generate a checkerboard pattern PDF for camera calibration.

    Args:
        square_size_mm: Size of each checkerboard square in millimeters
        rows: Number of internal corners vertically (default 6 for 7x10 board)
        cols: Number of internal corners horizontally (default 9 for 7x10 board)
        output_filename: Output PDF filename
    """
    # A4 paper dimensions in mm
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297

    # Calculate checkerboard dimensions
    # We need rows+1 and cols+1 squares for the specified internal corners
    board_width_mm = (cols + 1) * square_size_mm
    board_height_mm = (rows + 1) * square_size_mm

    # Check if checkerboard fits on A4 paper
    if board_width_mm > A4_WIDTH_MM or board_height_mm > A4_HEIGHT_MM:
        print(
            f"Warning: Checkerboard ({board_width_mm}x{board_height_mm} mm) is larger than A4 paper ({A4_WIDTH_MM}x{A4_HEIGHT_MM} mm)"
        )
        print("Consider using a smaller square size.")
        return False

    # Convert mm to inches (matplotlib uses inches)
    MM_TO_INCH = 1 / 25.4
    A4_WIDTH_INCH = A4_WIDTH_MM * MM_TO_INCH
    A4_HEIGHT_INCH = A4_HEIGHT_MM * MM_TO_INCH
    square_size_inch = square_size_mm * MM_TO_INCH

    # Create figure with exact A4 dimensions
    fig = plt.figure(figsize=(A4_WIDTH_INCH, A4_HEIGHT_INCH))
    ax = fig.add_subplot(111, aspect="equal")

    # Center the checkerboard on the page
    offset_x = (A4_WIDTH_MM - board_width_mm) / 2
    offset_y = (A4_HEIGHT_MM - board_height_mm) / 2

    # Draw checkerboard pattern
    for row in range(rows + 1):
        for col in range(cols + 1):
            # Alternate black and white squares
            if (row + col) % 2 == 0:
                color = "black"
            else:
                color = "white"

            # Calculate position in mm, then convert to inches
            x_mm = offset_x + col * square_size_mm
            y_mm = offset_y + (rows - row) * square_size_mm  # Flip y-axis
            x_inch = x_mm * MM_TO_INCH
            y_inch = y_mm * MM_TO_INCH

            # Draw square
            rect = patches.Rectangle(
                (x_inch, y_inch),
                square_size_inch,
                square_size_inch,
                linewidth=0,
                edgecolor="none",
                facecolor=color,
            )
            ax.add_patch(rect)

    # Set axis limits to A4 dimensions
    ax.set_xlim(0, A4_WIDTH_INCH)
    ax.set_ylim(0, A4_HEIGHT_INCH)
    ax.axis("off")

    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, output_filename)

    # Save as PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0, dpi=300)

    plt.close()

    print(f"Checkerboard PDF generated successfully!")
    print(f"  Output file: {output_path}")
    print(f"  Pattern: {rows}x{cols} internal corners ({rows+1}x{cols+1} squares)")
    print(f"  Square size: {square_size_mm} mm")
    print(f"  Board dimensions: {board_width_mm} x {board_height_mm} mm")
    print(f"\nPrint settings:")
    print(f"  - Print on A4 paper")
    print(f"  - Scale: 100% (NO SCALING)")
    print(f"  - Page orientation: Portrait")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate a checkerboard pattern PDF for camera calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python checkerboard_generator.py --size 25
  python checkerboard_generator.py --size 20 --rows 8 --cols 11
  python checkerboard_generator.py --size 30 --output my_checkerboard.pdf
        """,
    )
    parser.add_argument(
        "--size",
        type=float,
        required=True,
        help="Size of each checkerboard square in millimeters (e.g., 25)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of internal corner rows (default: 6, creates 7x10 board)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Number of internal corner columns (default: 9, creates 7x10 board)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkerboard.pdf",
        help="Output PDF filename (default: checkerboard.pdf)",
    )

    args = parser.parse_args()

    generate_checkerboard_pdf(args.size, args.rows, args.cols, args.output)


if __name__ == "__main__":
    main()
