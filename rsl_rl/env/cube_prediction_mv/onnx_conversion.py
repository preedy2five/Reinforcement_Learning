import torch
import torch.onnx
from model import CubePositionNet

def convert_to_onnx(model_path, onnx_path, input_size=(3, 224, 224)):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to the .pth model file
        onnx_path: Output path for the .onnx file
        input_size: Input image size (C, H, W)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = CubePositionNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create dummy input tensors (batch_size=1)
    dummy_imgA = torch.randn(1, *input_size).to(device)
    dummy_imgB = torch.randn(1, *input_size).to(device)
    dummy_imgC = torch.randn(1, *input_size).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # model being run
        (dummy_imgA, dummy_imgB, dummy_imgC),  # model input (tuple for multiple inputs)
        onnx_path,                      # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=11,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names=['imgA', 'imgB', 'imgC'],   # the model's input names
        output_names=['position'],      # the model's output names
        dynamic_axes={
            'imgA': {0: 'batch_size'},     # variable length axes
            'imgB': {0: 'batch_size'},
            'imgC': {0: 'batch_size'},
            'position': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully converted to ONNX format: {onnx_path}")

def verify_onnx_model(onnx_path):
    """
    Verify the exported ONNX model
    """
    import onnx
    import onnxruntime as ort
    
    # Load and check the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    # Test inference with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test inputs
    test_imgA = torch.randn(1, 3, 224, 224).numpy()
    test_imgB = torch.randn(1, 3, 224, 224).numpy()
    test_imgC = torch.randn(1, 3, 224, 224).numpy()
    
    # Run inference
    ort_inputs = {
        'imgA': test_imgA,
        'imgB': test_imgB,
        'imgC': test_imgC
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX model output shape: {ort_outputs[0].shape}")
    print(f"Sample output: {ort_outputs[0]}")

if __name__ == "__main__":
    # Convert the model
    model_path = "sv_tl_0.2236.pth"
    onnx_path = "sv_vl_0.1668.onnx"

    convert_to_onnx(model_path, onnx_path)
    
    # Optional: Verify the converted model
    try:
        verify_onnx_model(onnx_path)
    except ImportError:
        print("Install 'onnx' and 'onnxruntime' packages to verify the model")
        print("pip install onnx onnxruntime")