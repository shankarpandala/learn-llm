import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DreamBooth() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">DreamBooth: Personalizing Diffusion Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        DreamBooth fine-tunes the entire diffusion model to learn a specific subject from
        just 3-5 images. By binding the subject to a unique identifier token (e.g., "sks"),
        the model can generate novel images of that subject in diverse contexts, poses, and
        styles while preserving the subject's key visual features.
      </p>

      <DefinitionBlock
        title="DreamBooth"
        definition="DreamBooth fine-tunes a text-to-image diffusion model to associate a unique token identifier $V^*$ (e.g., 'sks') with a specific subject. The training objective is: $\mathcal{L} = \mathbb{E}_{t, \epsilon} [\| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, \text{'a photo of } V^* \text{ [class]'}) \|^2]$ where [class] is the subject's class (e.g., 'dog', 'person')."
        id="def-dreambooth"
      />

      <h2 className="text-2xl font-semibold">Prior Preservation Loss</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A critical component is the prior preservation loss that prevents catastrophic forgetting.
        The model generates class-specific images with the frozen original model, and uses these
        as regularization during fine-tuning.
      </p>
      <BlockMath math="\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruction}} + \lambda \cdot \mathcal{L}_{\text{prior}}" />

      <ExampleBlock
        title="DreamBooth Training Setup"
        problem="Fine-tune SD to learn your pet dog 'Rex' from 5 photos."
        steps={[
          { formula: '\\text{Identifier: } V^* = \\text{"sks"}, \\text{ Class: "dog"}', explanation: 'Choose a rare token as identifier, paired with the class noun.' },
          { formula: '\\text{Training prompts: "a photo of sks dog"}', explanation: 'All 5 training images use this caption.' },
          { formula: '\\text{Prior: Generate 200 images of "a photo of dog"}', explanation: 'Use the original model to create class regularization images.' },
          { formula: '\\text{Fine-tune 800-1200 steps, lr=5e-6}', explanation: 'Short fine-tuning with low learning rate to avoid overfitting.' },
        ]}
        id="example-dreambooth-setup"
      />

      <PythonCode
        title="dreambooth_training.py"
        code={`# DreamBooth training with diffusers
# Command-line approach (recommended):
# accelerate launch train_dreambooth.py \\
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
#   --instance_data_dir="./my_dog_photos" \\
#   --class_data_dir="./class_dog_photos" \\
#   --output_dir="./dreambooth_dog" \\
#   --instance_prompt="a photo of sks dog" \\
#   --class_prompt="a photo of dog" \\
#   --with_prior_preservation --prior_loss_weight=1.0 \\
#   --num_class_images=200 \\
#   --resolution=512 \\
#   --train_batch_size=1 \\
#   --gradient_accumulation_steps=1 \\
#   --learning_rate=5e-6 \\
#   --lr_scheduler="constant" \\
#   --max_train_steps=800 \\
#   --mixed_precision="fp16"

# Simplified DreamBooth training loop
import torch
import torch.nn.functional as F

def dreambooth_training_step(
    unet, vae, text_encoder, tokenizer, noise_scheduler,
    instance_batch, class_batch, instance_prompt, class_prompt,
    prior_loss_weight=1.0
):
    """Single DreamBooth training step with prior preservation."""

    def compute_loss(images, prompt):
        # Encode images to latent
        latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        # Text conditioning
        tokens = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
        text_emb = text_encoder(tokens.to(latents.device))[0]

        # Predict noise
        noise_pred = unet(noisy, timesteps, text_emb).sample
        return F.mse_loss(noise_pred, noise)

    # Instance loss (subject-specific)
    loss_instance = compute_loss(instance_batch, instance_prompt)

    # Prior preservation loss (class-specific)
    loss_prior = compute_loss(class_batch, class_prompt)

    # Combined loss
    total_loss = loss_instance + prior_loss_weight * loss_prior
    return total_loss, loss_instance.item(), loss_prior.item()

# Hyperparameter guidelines
print("DreamBooth Hyperparameters:")
configs = {
    "Learning rate": "1e-6 to 5e-6 (lower for faces)",
    "Training steps": "800-1200 (more steps = more overfitting)",
    "Instance images": "3-5 (more is better, up to ~20)",
    "Prior images": "200-300 per class",
    "Prior loss weight": "1.0 (standard)",
    "Resolution": "512 (SD 1.5) or 1024 (SDXL)",
}
for k, v in configs.items():
    print(f"  {k}: {v}")`}
        id="code-dreambooth"
      />

      <NoteBlock
        type="tip"
        title="DreamBooth Tips"
        content="Use diverse training images (different angles, lighting, backgrounds). Avoid images that are too similar. For human faces, 10-20 high-quality photos work best. The 'sks' identifier is commonly used, but any rare token works. Check results every 200 steps to catch overfitting early."
        id="note-dreambooth-tips"
      />

      <WarningBlock
        title="Overfitting and Language Drift"
        content="Without prior preservation, DreamBooth quickly overfits: the model can only generate the training images and loses diversity. Language drift is another risk where the class word ('dog') becomes synonymous with only your specific subject. Prior preservation loss mitigates both issues."
        id="warning-overfitting"
      />
    </div>
  )
}
