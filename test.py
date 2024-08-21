import torch
from feature_splat import rasterization

def main():
        device = 'cuda'
        means = torch.randn((100, 3), device=device)
        quats = torch.randn((100, 4), device=device)
        scales = torch.rand((100, 3), device=device) * 0.1
        colors = torch.zeros((100, 1024), device=device, requires_grad=True)
        opacities = torch.rand((100,), device=device)
        # define cameras
        viewmats = torch.eye(4, device=device)[None, :, :]
        Ks = torch.tensor([
           [300., 0., 100.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        width, height = 200, 200
        # render
        colors_render, alphas, meta = rasterization(
           means, quats, scales, opacities, colors, viewmats, Ks, width, height
        )
        loss_fn = torch.nn.CosineEmbeddingLoss()
        gt = torch.ones((200,200,1024), device = device).view(-1, 1024)
        target = torch.ones((200, 200), device=device).view(-1)
        loss = loss_fn(colors_render.unsqueeze(0).view(-1,1024), gt, target).mean()  # Assume the correct usage of CosineEmbeddingLoss
        print(loss)
        loss.backward()

        print(colors.grad)
        print(colors.grad.mean())

if __name__ == '__main__':
    main()