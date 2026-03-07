use cubecl::prelude::*;
use cubecl::{
    self,
    std::tensor::layout::{Coords1d, Layout, LayoutExpand},
};

#[derive(CubeType, Clone, Copy)]
/// Allows to work on the last dimension of the signal/spectrum (one window),
/// abstracting batches
pub struct BatchSignalLayout {
    num_samples: usize,
    stride_samples: usize,
    batch_offset: usize,
    #[cube(comptime)]
    line_size: usize,
}

#[cube]
impl BatchSignalLayout {
    pub fn new<F: Float>(tensor: &Tensor<Line<F>>, batch_index: usize) -> Self {
        let rank = tensor.rank();
        BatchSignalLayout {
            num_samples: tensor.shape(rank - 1),
            stride_samples: tensor.stride(rank - 1),
            batch_offset: batch_index * tensor.stride(rank - 2),
            line_size: tensor.line_size(),
        }
    }
}

#[cube]
impl Layout for BatchSignalLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> usize {
        (self.batch_offset + coords as usize * self.stride_samples) / self.line_size
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        self.num_samples
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.num_samples
    }
}
