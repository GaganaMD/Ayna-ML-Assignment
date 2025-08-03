import matplotlib.pyplot as plt

def show_triplet(input_image, color_name, output_gt, output_pred=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(input_image[0].cpu().numpy(), cmap='gray')
    plt.title('Input Polygon')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(output_gt.permute(1,2,0).cpu().numpy())
    plt.title(f'Target: {color_name}')
    plt.axis('off')
    if output_pred is not None:
        plt.subplot(1,3,3)
        plt.imshow(output_pred.permute(1,2,0).detach().cpu().numpy())
        plt.title('Predicted Output')
        plt.axis('off')
    plt.show()
