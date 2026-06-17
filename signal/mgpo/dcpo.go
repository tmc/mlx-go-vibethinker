package mgpo

import "fmt"

// PromptStats carries the cumulative per-prompt advantage history that DCPO's
// Smooth Advantage Standardization (SAS, arXiv 2509.02333) needs across training
// steps. It is keyed by a caller-supplied prompt identifier; each entry holds the
// smoothed advantage vector A_total from the prompt's previous visit and the
// visit count i.
//
// The zero value is an empty, ready-to-use store: the first time a prompt is
// seen (i becomes 1), SAS degenerates to the plain advantage, so a fresh store
// reproduces today's behavior exactly. Reuse one store across the steps of a run
// to accumulate the smoothing; a per-step store is equivalent to the baseline.
//
// PromptStats is not safe for concurrent use; SAS is applied at the
// advantage-assembly seam, which is single-threaded per training step.
type PromptStats struct {
	hist map[string]promptHistory
}

// promptHistory is one prompt's carried SAS state: the smoothed advantage from
// the last visit and the number of visits so far.
type promptHistory struct {
	total  []float64 // A_total: the smoothed advantage carried from the last visit
	visits int       // i: number of times this prompt has been standardized
}

// NewPromptStats returns an empty SAS history store. It is equivalent to the
// zero value and is provided for symmetry with the other constructors.
func NewPromptStats() *PromptStats {
	return &PromptStats{}
}

// smooth applies DCPO Smooth Advantage Standardization to one prompt group's
// freshly computed advantage aNew, using and updating the cumulative history for
// promptID. It returns the smoothed advantage of the same length.
//
// On the first visit (i becomes 1) A_total is initialized to aNew, so both
// smoothed estimates equal aNew and the result is exactly aNew — first visit ==
// plain advantage, the documented zero-value behavior. On later visits, with i
// the (post-increment) visit count, for each rollout slot:
//
//	SA_new   = ((i−1)/i)·A_new + (1/i)·A_total
//	SA_total = (1/i)·A_new   + ((i−1)/i)·A_total
//	A        = SA_new if |SA_new| < |SA_total| else SA_total
//
// The chosen A becomes the new A_total carried to the next visit. A_new is the
// freshly standardized group advantage (std-normalized or Dr.GRPO no-std,
// whichever the options selected) — SAS smooths whatever advantage it is given,
// so it composes with Dr.GRPO. w_ME is applied by the caller after this, so the
// MGPO no-op rule is preserved.
func (s *PromptStats) smooth(promptID string, aNew []float64) ([]float64, error) {
	if s.hist == nil {
		s.hist = make(map[string]promptHistory)
	}
	h := s.hist[promptID]
	if h.visits > 0 && len(h.total) != len(aNew) {
		return nil, fmt.Errorf("mgpo: DCPO-SAS prompt %q group size changed from %d to %d", promptID, len(h.total), len(aNew))
	}
	i := h.visits + 1
	out := make([]float64, len(aNew))
	if i == 1 {
		// First visit: A_total := A_new, so SA_new = SA_total = A_new.
		copy(out, aNew)
		s.hist[promptID] = promptHistory{total: append([]float64(nil), out...), visits: i}
		return out, nil
	}
	fi := float64(i)
	wOld := float64(i-1) / fi // (i−1)/i
	wNew := 1.0 / fi          // 1/i
	for j, an := range aNew {
		at := h.total[j]
		saNew := wOld*an + wNew*at
		saTotal := wNew*an + wOld*at
		if abs(saNew) < abs(saTotal) {
			out[j] = saNew
		} else {
			out[j] = saTotal
		}
	}
	s.hist[promptID] = promptHistory{total: append([]float64(nil), out...), visits: i}
	return out, nil
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
