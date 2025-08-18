from PIL import Image, ImageOps

def black_to_white_logo(input_path, output_path):
    # Open the image
    img = Image.open(input_path).convert("RGBA")

    # Split into channels
    r, g, b, a = img.split()

    # Convert RGB to grayscale
    gray = ImageOps.grayscale(img)

    # Invert grayscale (black → white, white → black)
    inverted = ImageOps.invert(gray)

    # Put alpha back so transparency is preserved
    final_img = Image.merge("RGBA", (inverted, inverted, inverted, a))

    # Save result
    final_img.save(output_path, "PNG")
    print(f"Saved white logo to {output_path}")

# Example usage
black_to_white_logo("assets\shelflytics_logo_no_word_transparent.png", "assets\shelflytics_logo_no_word_transparent_white.png")
black_to_white_logo("assets\shelflytics_logo_transparent.png", "assets\shelflytics_logo_transparent_white.png")
