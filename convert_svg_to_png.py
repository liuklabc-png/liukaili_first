import cairosvg

# Convert SVG to PNG
svg_path = 'frontend/public/logo.svg'
png_path = 'frontend/public/logo.png'

# Convert the SVG to PNG with a specific width and height
cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=256, output_height=256)

print(f"Converted {svg_path} to {png_path}")