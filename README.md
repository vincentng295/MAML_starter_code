# MAML starter code

Đây là project triển khai Model-Agnostic Meta-Learning (MAML) trên tập dữ liệu Omniglot. Project cho phép huấn luyện mô hình MAML để nhanh chóng thích nghi với các tác vụ mới.

## Cấu trúc thư mục

Project có cấu trúc cơ bản như sau:

```
MAML_starter_code/
├── .gitattributes
├── .gitignore
├── __init__.py
├── data_generator.py
├── environment.yml
├── environment_cuda.yml
├── grader.py
├── graderUtil.py
├── learner.py
├── meta.json
├── omniglot.py
├── points.json
├── requirements.txt
├── run_bio.sh
├── run_metabolism.py
└── submission/
    ├── .DS_Store
    ├── __init__.py
    ├── maml.py
    └── omniglot_resized/ (Thư mục dataset Omniglot)
        └── Alphabet_of_the_Magi/
            └── character01/
                └── 0709_01.png
                └── ...
```

**Lưu ý quan trọng:** Thư mục `omniglot_resized` chứa dữ liệu Omniglot phải nằm trong thư mục `submission`.

## Yêu cầu

*   Python 3.9+
*   pip
*   Các thư viện được liệt kê trong `requirements.txt`
*   (Tùy chọn) CUDA Toolkit nếu muốn sử dụng GPU với PyTorch.

## Cài đặt

1.  **Clone repository:**

    ```bash
    git clone https://github.com/vincentng295/MAML_starter_code.git
    cd MAML_starter_code
    ```

2.  **Tạo môi trường ảo và cài đặt dependencies:**

    *   **Cách 1 (khuyên dùng - nếu có Anaconda/Miniconda):**
        ```bash
        conda env create -f environment.yml
        conda activate CS330
        ```

    *   **Cách 2 (không có Anaconda):**
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # Trên Windows dùng `venv\Scripts\activate`
        
        # Tạo file requirements.txt chỉ với các gói cần thiết
        echo -e "torch\ntorchvision\ntorchaudio\ncloudpickle\ncycler\nh5py\nlockfile\nopt-einsum\npackaging\npandas\npickleshare\npython-dateutil\nrequests\nscipy\nsix\nscikit-learn\ntensorboard\npytest\nimageio\nsvglib\ngoogledrivedownloader\nmatplotlib\ntimeout_decorator\nrouge-score\nhuggingface-hub\ntransformers\ndatasets\nnumpy" > requirements.txt

        # Cài đặt PyTorch CPU:
        pip install -r requirements.txt
        # Hoặc cài đặt PyTorch GPU (kiểm tra phiên bản CUDA của bạn, ví dụ cu118):
        # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

3.  **Tải và giải nén Dataset Omniglot:**

    Dữ liệu Omniglot sẽ được tự động tải về khi chạy script lần đầu với `--cache` hoặc qua `gdd.download_file_from_google_drive` trong `omniglot.py`. Đảm bảo file `omniglot_resized.zip` được giải nén vào `submission/omniglot_resized`.

## Chạy Project

Để chạy MAML, sử dụng lệnh sau:

*   **Chạy với CPU:**

    ```bash
    export PYTHONPATH=$(pwd) && python3 submission/maml.py --num_workers 0
    ```

*   **Chạy với GPU (nếu đã cài đặt PyTorch CUDA):**

    ```bash
    export PYTHONPATH=$(pwd) && python3 submission/maml.py --device gpu --num_workers 0
    ```

*   **Giảm `batch_size` để debug (ví dụ):**

    ```bash
    export PYTHONPATH=$(pwd) && python3 submission/maml.py --device gpu --num_workers 0 --batch_size 1
    ```

## Các đối số tùy chỉnh

Bạn có thể điều chỉnh các đối số sau khi chạy `maml.py`:

*   `--log_dir`: Thư mục để lưu log và checkpoint.
*   `--num_way`: Số lượng lớp trong một tác vụ (mặc định: 5).
*   `--num_support`: Số lượng ví dụ hỗ trợ trên mỗi lớp (mặc định: 1).
*   `--num_query`: Số lượng ví dụ truy vấn trên mỗi lớp (mặc định: 15).
*   `--num_inner_steps`: Số bước tối ưu hóa inner-loop (mặc định: 1).
*   `--inner_lr`: Tốc độ học inner-loop (mặc định: 0.4).
*   `--learn_inner_lrs`: Có học tốc độ học inner-loop không (mặc định: False).
*   `--outer_lr`: Tốc độ học outer-loop (mặc định: 0.001).
*   `--batch_size`: Số lượng tác vụ trên mỗi lần cập nhật outer-loop (mặc định: 16).
*   `--num_train_iterations`: Số lần cập nhật outer-loop để huấn luyện (mặc định: 15000).
*   `--test`: Chạy chế độ kiểm thử thay vì huấn luyện (mặc định: False).
*   `--checkpoint_step`: Bước checkpoint để tải lại (mặc định: -1, bỏ qua).
*   `--num_workers`: Số lượng worker cho DataLoader (mặc định: 2).
*   `--cache`: Tải xuống và giải nén dataset (mặc định: False).
*   `--device`: Chọn thiết bị ('cpu' hoặc 'gpu') (mặc định: 'cpu').
