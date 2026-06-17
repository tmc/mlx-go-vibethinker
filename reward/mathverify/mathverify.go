package mathverify

import (
	"math"
	"regexp"
	"strconv"
	"strings"

	rl "github.com/tmc/mlx-go/examples/mlx-go-rl"
)

// epsilon is the tolerance for decimal equivalence of two normalized answers.
const epsilon = 1e-9

// ExtractAnswer returns the final answer in text: the contents of the last
// \boxed{...} if any is present, otherwise the last number-or-expression token
// on the trailing line. It returns ("", false) when no answer can be found.
func ExtractAnswer(text string) (string, bool) {
	return extractAnswer(text)
}

// Normalize canonicalizes an answer string for comparison: it strips $, \text,
// and \boxed wrappers, removes whitespace and thousands-commas, and trims
// trailing zeros from decimals. The result is the string compared character by
// character when the two answers are not both numeric.
func Normalize(answer string) string {
	return normalize(answer)
}

// Equivalent reports whether got and want denote the same final answer after
// normalization. Numeric answers (including simple a/b fractions) compare by
// value within a small tolerance; otherwise the normalized strings must match.
func Equivalent(got, want string) bool {
	return equivalent(got, want)
}

// Reward returns the binary final-answer reward for a completion against a
// reference answer: 1 when the completion's extracted answer is equivalent to
// the reference, 0 otherwise (including a missing answer on either side).
func Reward(completion, reference string) float64 {
	return reward(completion, reference)
}

// Verify is a rl.VerifyFunc closure factory: VerifyFor(reference) returns a
// VerifyFunc that scores a completion against the fixed reference answer,
// reporting the extracted answer as feedback. Use Environment to obtain the
// adapted rl.RichEnvironment.
func VerifyFor(reference string) rl.VerifyFunc {
	return func(prompt, completion string) (float64, string, error) {
		score := reward(completion, reference)
		ans, ok := extractAnswer(completion)
		if !ok {
			return score, "no final answer found", nil
		}
		if score == 1 {
			return score, "", nil
		}
		return score, "answer " + ans + " not equivalent to " + reference, nil
	}
}

// Environment adapts the final-answer reward for a fixed reference answer to an
// rl.RichEnvironment, so it composes with the MGPO/GRPO optimizer and the rest
// of the reward pipeline. The prompt is ignored; only the completion's answer
// is scored against reference.
func Environment(reference string) rl.RichEnvironment {
	return rl.EnvFromVerifyFunc(VerifyFor(reference))
}

// --- cores ---

func reward(completion, reference string) float64 {
	got, okG := extractAnswer(completion)
	want, okW := extractAnswer(reference)
	if !okG || !okW {
		return 0
	}
	if equivalent(got, want) {
		return 1
	}
	return 0
}

func equivalent(got, want string) bool {
	ng, nw := normalize(got), normalize(want)
	if ng == "" || nw == "" {
		return false
	}
	if vg, okG := numericValue(ng); okG {
		if vw, okW := numericValue(nw); okW {
			return math.Abs(vg-vw) <= epsilon*(1+math.Abs(vw))
		}
		return false
	}
	// One side is non-numeric: fall back to exact normalized-string match.
	return ng == nw
}

var (
	// boxedRe matches \boxed or \fbox; the brace body is balanced by hand.
	boxedRe = regexp.MustCompile(`\\(?:boxed|fbox)\s*{`)
	// numberRe matches a trailing signed number or simple fraction.
	numberRe = regexp.MustCompile(`-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*(?:\.\d+)?)?`)
)

func extractAnswer(text string) (string, bool) {
	text = strings.TrimSpace(text)
	if text == "" {
		return "", false
	}
	if body, ok := lastBoxed(text); ok {
		body = strings.TrimSpace(body)
		if body != "" {
			return body, true
		}
	}
	// No boxed answer: take the last numeric/fraction token on the trailing
	// non-empty line.
	lines := strings.Split(text, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		if m := numberRe.FindAllString(line, -1); len(m) > 0 {
			return strings.TrimSpace(m[len(m)-1]), true
		}
		// A non-empty trailing line with no number: use it verbatim so that
		// symbolic answers (e.g. a variable name) still compare.
		return line, true
	}
	return "", false
}

// lastBoxed returns the brace-balanced body of the last \boxed{...}/\fbox{...}
// in text. It scans all matches and returns the final one's body.
func lastBoxed(text string) (string, bool) {
	locs := boxedRe.FindAllStringIndex(text, -1)
	if locs == nil {
		return "", false
	}
	loc := locs[len(locs)-1]
	// loc[1]-1 is the opening '{'.
	depth := 0
	for i := loc[1] - 1; i < len(text); i++ {
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[loc[1]:i], true
			}
		}
	}
	return "", false // unbalanced
}

// normalize canonicalizes an answer string for comparison.
func normalize(answer string) string {
	s := answer
	// Strip \boxed/\fbox wrappers that may survive extraction of a reference.
	if body, ok := lastBoxed(s); ok {
		s = body
	}
	// Strip \text{...} wrappers, keeping their body.
	s = stripTextWrap(s)
	// Drop LaTeX dollar math delimiters and brace/escape artifacts.
	s = strings.ReplaceAll(s, "$", "")
	s = strings.ReplaceAll(s, "\\!", "")
	s = strings.ReplaceAll(s, "\\,", "")
	s = strings.ReplaceAll(s, "\\ ", "")
	s = strings.ReplaceAll(s, "{,}", ",") // LaTeX thousands separator
	s = strings.ReplaceAll(s, "{", "")
	s = strings.ReplaceAll(s, "}", "")
	// Remove all whitespace.
	s = strings.Join(strings.Fields(s), "")
	// Remove thousands-commas only between digits.
	s = stripDigitCommas(s)
	// Strip a single trailing period (sentence punctuation), but not a decimal
	// point inside a number — handled by trimming trailing zeros below.
	s = strings.TrimSpace(s)
	s = trimDecimalZeros(s)
	return s
}

var textWrapRe = regexp.MustCompile(`\\(?:text|mathrm|mbox)\s*{([^{}]*)}`)

func stripTextWrap(s string) string {
	for {
		out := textWrapRe.ReplaceAllString(s, "$1")
		if out == s {
			return out
		}
		s = out
	}
}

// stripDigitCommas removes commas that sit between two digits (thousands
// separators) while preserving commas used elsewhere.
func stripDigitCommas(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ',' && i > 0 && i+1 < len(s) && isDigit(s[i-1]) && isDigit(s[i+1]) {
			continue
		}
		b.WriteByte(c)
	}
	return b.String()
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// trimDecimalZeros trims trailing zeros (and a dangling decimal point) from a
// plain decimal literal, so "12.0" → "12" and "0.50" → "0.5". Non-decimal
// strings are returned unchanged.
func trimDecimalZeros(s string) string {
	if !strings.Contains(s, ".") {
		return s
	}
	// Only operate when the whole string is a decimal number (optional sign).
	body := s
	if strings.HasPrefix(body, "-") || strings.HasPrefix(body, "+") {
		body = body[1:]
	}
	for i := 0; i < len(body); i++ {
		if !isDigit(body[i]) && body[i] != '.' {
			return s // not a plain decimal; leave it alone
		}
	}
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return s
}

// numericValue parses a normalized answer as a number or simple a/b fraction,
// returning its float value. It reports false when the string is not numeric.
func numericValue(s string) (float64, bool) {
	if s == "" {
		return 0, false
	}
	if i := strings.IndexByte(s, '/'); i >= 0 {
		num, err1 := strconv.ParseFloat(s[:i], 64)
		den, err2 := strconv.ParseFloat(s[i+1:], 64)
		if err1 != nil || err2 != nil || den == 0 {
			return 0, false
		}
		return num / den, true
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// ensure Environment satisfies the interface at compile time.
var _ rl.RichEnvironment = Environment("0")
