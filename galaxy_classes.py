from typing import Literal, overload
from abc import ABC

#Galaxy Classes
class Galaxy(ABC):
    def __init__(
            self,
            morphology: Literal["disk"],
            sampling: Literal["equal-mass", "importance"],
            N_particles: int,
            M_tot: float = None,
            seed: int = None,
    ):
        if M_tot is None:
            M_tot = N_particles
        self.morphology = morphology        # Type of galaxy
        self.sampling = sampling            # Sampling method
        self.seed = seed                    # Seed for reproducibility
        self.N_particles = N_particles      # Total number of particles
        self.M_tot = M_tot                  # Total mass of the galaxy

    def __str__(self) -> str:
        return f"{self.morphology}_{self.sampling}_{self.seed}"

class DiskGalaxy(Galaxy):
    def __init__(
            self,
            morphology: Literal["disk"],
            sampling: Literal["equal-mass", "importance"],
            N_particles: int,
            Rd = float,
            z0 = float,
            M_tot: float = None,
            seed: int = None,
    ):
        super().__init__(morphology, sampling, N_particles, M_tot, seed)
        self.Rd = Rd                        # Radial scale length of the stellar disk: Σ(R) = Σ0 exp(−R/Rd)
                                            # Larger Rd extends the disk and lowers centra surface density
        self.z0 = z0                        # Vertical scale height: ρ(z) ∝ exp(−|z|/z0)
                                            # Larger z0 thickens the disk
    def galaxy_type(self):
        return self.morphology

@overload
def generate_galaxy(
    morphology: Literal["disk"],
    sampling: Literal["equal-mass", "importance"],
    N_particles: int,
    Rd: float,
    z0: float,
    M_tot: float = None,
    seed: int = None,
) -> DiskGalaxy: ...

def generate_galaxy(
        morphology: Literal["disk"],
        sampling: Literal["equal-mass", "importance"],
        N_particles: int,
        Rd: float | None = None,
        z0: float | None = None,
        M_tot: float | None = None,
        seed: int | None = None,
):
    if morphology == "disk":
        return DiskGalaxy("disk", sampling, N_particles, Rd, z0, M_tot, seed)
    raise ValueError(f"Unknown morphology: {morphology!r}")

gal = generate_galaxy("disk", "equal-mass", 10000, 10, 1, seed=42)
print(gal)