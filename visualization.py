from mayavi import mlab
import os
import torch.nn.functional as F
import imageio

def plot_graph_mayavi(data, model, layer, molecule_name):
    mlab.clf()
    model.eval()
    x, edge_index = data.x, data.edge_index
    for i in range(layer):
        x = model.convs[i](x, edge_index)
        x = F.relu(x)
        if i < layer - 1:
            x = F.dropout(x, training=model.training)
    
    colors = x.sum(dim=1).detach().numpy()

    pts = mlab.points3d(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2], colors, colormap='viridis', scale_factor=0.3)
    for i, j in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()):
        mlab.plot3d([data.pos[i, 0], data.pos[j, 0]], 
                    [data.pos[i, 1], data.pos[j, 1]], 
                    [data.pos[i, 2], data.pos[j, 2]], color=(0, 0, 0), tube_radius=0.05)

    mlab.text(0.5, 0.95, f'{molecule_name} Layer: {layer}', width=1, color=(1, 1, 1))
    mlab.colorbar(pts, title="Color Scale", orientation='vertical')

def save_frames(inputs, model, molecule_names, folder="frames"):
    mlab.options.offscreen = True

    if not os.path.exists(folder):
        os.makedirs(folder)

    f = mlab.gcf()

    frame_idx = 0
    for idx, data in enumerate(inputs):
        for layer in range(1, len(model.convs) + 1):
            plot_graph_mayavi(data, model, layer, molecule_names[idx])
            mlab.savefig(os.path.join(folder, f"frame_{frame_idx:04}.png"), size=(1920, 1080))
            frame_idx += 1

    mlab.close()

def frames_to_video(input_folder="frames", output_video="output.mp4", fps=0.5):
    images = [img for img in sorted(os.listdir(input_folder)) if img.endswith(".png")]
    frame_array = []

    for i in range(len(images)):
        img_path = os.path.join(input_folder, images[i])
        frame_array.append(imageio.imread(img_path))

    # Convert frames to video
    imageio.mimsave(output_video, frame_array, fps=fps)
