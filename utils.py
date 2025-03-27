
# Allowed image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Split the menu text into smaller chunks for processing
def split_menu_text(menu_text: str, chunk_size: int = 12):
    lines = [line.strip() for line in menu_text.splitlines() if line.strip()]
    return ["\n".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]
