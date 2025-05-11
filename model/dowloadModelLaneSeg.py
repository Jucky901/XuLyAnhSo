# Mô hình không nằm trong repo vì quá 100MB.
# Tải model từ:
# https://drive.google.com/file/d/1KEB3GUvJF8sM3RoNc8yv9ZCPUtlsnv7L/view?usp=sharing
import gdown
file_id = "1KEB3GUvJF8sM3RoNc8yv9ZCPUtlsnv7L"
url = f"https://drive.google.com/uc?id={file_id}"

output = "model/lanedetectionSegmentation.pth"
gdown.download(url, output, quiet=False)
#