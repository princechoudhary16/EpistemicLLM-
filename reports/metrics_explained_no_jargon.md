# What "metrics.py" does — explained with no tech or math background needed

Think of the AI model as a student taking a quiz, over and over, thousands of times.
This file is the **grading rulebook** the team uses to mark that student fairly — and,
just as importantly, to catch a student who's gaming the test.

## The four things a student can do with a question

For every question, the AI does one of four things:

1. **Gets it right.**
2. **Gets it wrong** — but answers confidently, as if it's sure.
3. **Says "I don't know."**
4. Its answer comes out **broken or unreadable** (a technical glitch, not really an
   attempt at all — like a student's pencil breaking mid-answer). This gets set aside
   and doesn't count against or for the student.

The whole point of this project is that #2 — confidently wrong — is the dangerous one.
A student who says "I don't know" when unsure is at least being honest. A student who
makes something up and says it with total confidence is the one who gets people into
trouble, because you can't tell the difference just by looking at how sure they sound.

## Why you can't just count "how many did it get right"

Here's the trap this file is specifically built to avoid:

Imagine a student who refuses to answer 90% of the quiz, saying "I don't know" to
almost everything, but gets every question they *do* attempt correct. If you only look
at "accuracy," this student looks amazing. But they're actually useless — they've just
learned to dodge anything they're not 100% sure about.

So instead of one number, this rulebook always reports several numbers together:

- **How often it's right** (out of everything)
- **How often it's confidently wrong** (the dangerous kind of mistake)
- **How often it says "I don't know"**
- **How often it even tries to answer at all** (instead of dodging)
- **Of the times it did try, how often it was actually right**

Reporting all of these side by side means nobody can make a model look good just by
teaching it to refuse more often — the "how often it even tries" number would expose that
immediately.

## Asking the same question more than once

For some questions, the AI is asked the same thing five separate times, not just once.
Why? Because it reveals something a single answer can't: does the model actually *know*
the fact, or did it just get lucky/unlucky one time?

- If it gets the same question right all 5 times → it really knows this.
- If it gets it right sometimes and wrong other times → it's guessing, not knowing.
- If it's wrong all 5 times → it doesn't know this at all.

This is like asking a student the same question on five different days instead of just
once — one lucky guess on a single day would never tell you that.

## Checking whether its confidence can be trusted

Imagine a weather forecaster who says "90% chance of rain" — if that's meaningful, it
should actually rain about 90% of the time whenever they say that. If they say "90%"
and it only rains 40% of the time, their confidence number is meaningless — worse than
meaningless, because people are trusting a number that lies to them.

This file checks the exact same thing for the AI: when it seems very confident, is it
actually right that often? If not, its confidence is misleading, and that's exactly the
kind of "confidently wrong" behavior this whole project is trying to catch and reduce.

## "What if we only let it answer when it's really sure?"

One idea for making an AI safer is: let it answer freely when it's confident, but make it
say "I don't know" whenever it's the least bit unsure. This file has a tool that tests
exactly that idea — it tries out different "how sure does it need to be" cutoffs, and for
each one, measures two things that trade off against each other:

- **How many questions does it still answer** (turn the cutoff up too high, and it
  answers almost nothing — not useful)
- **Of the ones it does answer, how often is it right** (turn the cutoff down too low,
  and it answers everything but makes lots of confident mistakes)

This produces a whole curve of options rather than one number, because there's no single
"correct" tradeoff — a doctor's assistant AI and a trivia game AI would want very
different cutoffs.

## "Is this result actually real, or could it just be luck?"

Say you test two different versions of the AI on 100 questions, and version A scores 62%
while version B scores 65%. Is version B actually better, or is that 3-point gap just
random noise — the kind of difference you'd see even between two identical students on
two different days?

This file has a method for answering that, and it works like this: imagine you had a
giant bag containing all 100 questions, and you scooped out 100 questions *with your eyes
closed*, allowing yourself to grab the same question more than once and skip others
entirely — like drawing raffle tickets from a bag and putting each one back before the
next draw. You score the AI on this "shuffled" set of 100, write down the score, then put
everything back and do the whole thing again. And again. A thousand times.

If version B is *genuinely* better, it'll keep winning across almost all of those
thousand reshuffles — a small, occasional loss here or there is fine, but the overall
pattern holds up. If the 3-point gap was really just luck, though, sometimes A will win
and sometimes B will win across those thousand tries, roughly at random — meaning the
original result doesn't actually mean anything reliable.

This matters especially because some of the questions get asked five times each, not
once — so this method is careful to always keep a question's five answers bundled
together when reshuffling, the same way you'd keep a family together rather than
splitting them up randomly, since those five answers aren't really five separate,
independent opinions — they're five looks at the same one thing.

## Why any of this matters

Six people on this team will each try their own idea for making the AI more honest about
what it does and doesn't know. Without a shared, fair rulebook like this one, it would be
very easy — even by accident — to make one idea look better than another just by grading
it more generously, or by not noticing that "low mistakes" secretly meant "refuses to
answer almost everything." This file exists so that whichever idea genuinely works best
can actually be shown to work best, honestly.
