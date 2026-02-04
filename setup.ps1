# Create necessary directories
$directories = @(
    "data/raw",
    "data/processed",
    "data/samples",
    "outputs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# Check if .env exists
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from template - please edit with your Azure credentials"
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate and install dependencies
Write-Host "Activating virtual environment and installing dependencies..."
& ".\venv\Scripts\Activate.ps1"
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete! Next steps:"
Write-Host "1. Edit .env with your Azure credentials"
Write-Host "2. Run 'az login' to authenticate"
Write-Host "3. Run 'python main.py prepare --sample' to create sample data"
Write-Host "4. Run 'python main.py train' to start fine-tuning"
