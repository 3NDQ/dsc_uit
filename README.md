- Sử dụng kaggle
- Tạo 1 kaggle notebook train rồi import finetune_dsc.ipynb vào
- Tạo thêm 1 kaggle notebook test rồi import predict.ipynb vào
- Sau khi khạy kaggle notebook train sẽ lưu các model được finetune ở từng epoch, quan sát log để chọn model tốt nhất.
- Vào kaggle notebook test chỗ add input -> yourwork (xếp lại theo ngày) -> chọn kaggle note book train vừa chạy xong
- Dán các model tốt nhất mà bạn chọn vào model_path để tiến hành test và trả kết quả, sau khi chạy xong sẽ xuất hiện các file result{x}.json 
