from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_heatmap(model, input_tensor, image_np):
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor)[0, :]
    heatmap = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
    return heatmap