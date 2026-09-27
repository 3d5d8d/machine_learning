from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class HessianResult:
    eigenvalues: np.ndarray
    max_eigenvalue: float
    max_eigenvector: torch.Tensor
    parameter_names: list[str]


def collect_fixed_batches(data_loader, device, num_batches=5):
    """Collect fixed batches so every landscape point uses identical data."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    batches = []
    for batch_index, (inputs, targets) in enumerate(data_loader):
        if batch_index >= num_batches:
            break

        batches.append((inputs.to(device), targets.to(device)))

    if not batches:
        raise ValueError("data_loader did not provide any batches")

    return batches


def snapshot_parameters(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def create_random_direction(model, normalization="parameter"):
    """Create a random direction with scale comparable to the model parameters."""
    parameters = snapshot_parameters(model)
    direction = {
        name: torch.randn_like(parameter)
        for name, parameter in parameters.items()
    }

    if normalization == "parameter":
        for name, parameter in parameters.items():
            parameter_norm = torch.linalg.vector_norm(parameter)
            direction_norm = torch.linalg.vector_norm(direction[name])

            if parameter_norm > 0 and direction_norm > 0:
                direction[name].mul_(parameter_norm / direction_norm)
            else:
                direction[name].zero_()

    elif normalization == "global":
        parameter_norm = _dictionary_norm(parameters)
        direction_norm = _dictionary_norm(direction)

        if parameter_norm > 0 and direction_norm > 0:
            scale = parameter_norm / direction_norm
            direction = {
                name: value * scale
                for name, value in direction.items()
            }
    else:
        raise ValueError(
            "normalization must be either 'parameter' or 'global'"
        )

    return direction


def orthogonalize_direction(direction, reference):
    """Make direction globally orthogonal to reference without changing its norm."""
    direction_norm = _dictionary_norm(direction)
    reference_norm_squared = _dictionary_dot(reference, reference)

    if reference_norm_squared <= 0:
        raise ValueError("reference direction has zero norm")

    projection = _dictionary_dot(direction, reference) / reference_norm_squared
    orthogonal = {
        name: direction[name] - projection * reference[name]
        for name in direction
    }

    orthogonal_norm = _dictionary_norm(orthogonal)
    if orthogonal_norm <= 0:
        raise ValueError("failed to create an orthogonal direction")

    scale = direction_norm / orthogonal_norm
    return {
        name: value * scale
        for name, value in orthogonal.items()
    }


@torch.no_grad()
def compute_1d_loss_landscape(
    model,
    batches,
    direction,
    radius=0.5,
    num_points=41,
):
    """Evaluate L(theta + alpha * direction) on a fixed set of batches."""
    if num_points < 3 or num_points % 2 == 0:
        raise ValueError("num_points must be an odd integer greater than or equal to 3")

    was_training = model.training
    model.eval()

    base_parameters = snapshot_parameters(model)
    alphas = torch.linspace(-radius, radius, num_points)
    losses = []

    try:
        for alpha in tqdm(alphas, desc="1D loss landscape"):
            perturbed_parameters = {
                name: base_parameters[name] + alpha.item() * direction[name]
                for name in base_parameters
            }
            losses.append(
                _mean_functional_loss(model, perturbed_parameters, batches)
            )
    finally:
        model.train(was_training)

    return alphas.numpy(), np.asarray(losses)


@torch.no_grad()
def compute_2d_loss_landscape(
    model,
    batches,
    direction_x,
    direction_y,
    radius=0.5,
    num_points=21,
):
    """Evaluate a two-direction slice of the high-dimensional loss surface."""
    if num_points < 3 or num_points % 2 == 0:
        raise ValueError("num_points must be an odd integer greater than or equal to 3")

    was_training = model.training
    model.eval()

    base_parameters = snapshot_parameters(model)
    coordinates = torch.linspace(-radius, radius, num_points)
    loss_grid = np.empty((num_points, num_points), dtype=np.float64)

    try:
        for row, beta in enumerate(tqdm(coordinates, desc="2D loss landscape")):
            for column, alpha in enumerate(coordinates):
                perturbed_parameters = {
                    name: (
                        base_parameters[name]
                        + alpha.item() * direction_x[name]
                        + beta.item() * direction_y[name]
                    )
                    for name in base_parameters
                }
                loss_grid[row, column] = _mean_functional_loss(
                    model,
                    perturbed_parameters,
                    batches,
                )
    finally:
        model.train(was_training)

    coordinate_values = coordinates.numpy()
    return coordinate_values, coordinate_values.copy(), loss_grid


def analyze_hessian_spectrum(
    model,
    batches,
    num_steps=10,
    target_name=None,
):
    """Approximate Hessian eigenvalues with Lanczos iterations."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (target_name is None or target_name in name)
    ]

    if not named_parameters:
        raise ValueError(f"no parameters matched target_name={target_name!r}")

    parameter_names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    num_parameters = sum(parameter.numel() for parameter in parameters)

    was_training = model.training
    model.eval()

    q = torch.randn(
        num_parameters,
        device=parameters[0].device,
        dtype=parameters[0].dtype,
    )
    q /= torch.linalg.vector_norm(q)

    q_previous = torch.zeros_like(q)
    beta_previous = torch.zeros((), device=q.device, dtype=q.dtype)
    basis = []
    alphas = []
    betas = []

    try:
        for step in tqdm(range(num_steps), desc="Lanczos Hessian"):
            basis.append(q)
            w = _mean_hessian_vector_product(
                model,
                batches,
                parameters,
                q,
            )

            if step > 0:
                w = w - beta_previous * q_previous

            alpha = torch.dot(q, w)
            alphas.append(alpha)
            w = w - alpha * q

            # Full re-orthogonalization improves numerical stability.
            for basis_vector in basis[:-1]:
                w = w - torch.dot(basis_vector, w) * basis_vector

            beta = torch.linalg.vector_norm(w)
            if beta <= 1e-10 or step == num_steps - 1:
                break

            betas.append(beta)
            q_previous = q
            q = w / beta
            beta_previous = beta
    finally:
        model.train(was_training)

    actual_steps = len(alphas)
    tridiagonal = torch.zeros(
        actual_steps,
        actual_steps,
        device=q.device,
        dtype=q.dtype,
    )
    tridiagonal.diagonal().copy_(torch.stack(alphas))

    if actual_steps > 1:
        off_diagonal = torch.stack(betas[: actual_steps - 1])
        tridiagonal.diagonal(1).copy_(off_diagonal)
        tridiagonal.diagonal(-1).copy_(off_diagonal)

    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    maximum_index = torch.argmax(eigenvalues)
    maximum_coefficients = eigenvectors[:, maximum_index]

    maximum_eigenvector = torch.zeros_like(basis[0])
    for index in range(actual_steps):
        maximum_eigenvector += maximum_coefficients[index] * basis[index]
    maximum_eigenvector /= torch.linalg.vector_norm(maximum_eigenvector)

    return HessianResult(
        eigenvalues=eigenvalues.detach().cpu().numpy(),
        max_eigenvalue=float(eigenvalues[maximum_index].detach().cpu()),
        max_eigenvector=maximum_eigenvector.detach().cpu(),
        parameter_names=parameter_names,
    )


def _mean_functional_loss(model, parameters, batches):
    total_loss = 0.0

    for inputs, targets in batches:
        _, loss = torch.func.functional_call(
            model,
            parameters,
            (inputs, targets),
        )
        total_loss += loss.item()

    return total_loss / len(batches)


def _mean_hessian_vector_product(model, batches, parameters, vector):
    vector_parts = _unflatten_vector(vector, parameters)
    result = torch.zeros_like(vector)

    for inputs, targets in batches:
        model.zero_grad(set_to_none=True)

        with _math_attention_context(inputs.device):
            _, loss = model(inputs, targets)
            gradients = torch.autograd.grad(
                loss,
                parameters,
                create_graph=True,
            )
            gradient_dot_vector = sum(
                torch.sum(gradient * vector_part)
                for gradient, vector_part in zip(gradients, vector_parts)
            )
            hessian_vector = torch.autograd.grad(
                gradient_dot_vector,
                parameters,
            )

        result += torch.cat(
            [value.contiguous().view(-1) for value in hessian_vector]
        ).detach()

    return result / len(batches)


def _unflatten_vector(vector, parameters):
    values = []
    offset = 0

    for parameter in parameters:
        numel = parameter.numel()
        values.append(vector[offset:offset + numel].view_as(parameter))
        offset += numel

    if offset != vector.numel():
        raise ValueError("vector size does not match selected parameters")

    return values


def _dictionary_dot(left, right):
    return sum(
        torch.sum(left[name] * right[name])
        for name in left
    )


def _dictionary_norm(values):
    return torch.sqrt(_dictionary_dot(values, values))


def _math_attention_context(device):
    if device.type != "cuda":
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
