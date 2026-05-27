# arXiv Endorsement Email — Drafts

You need ONE endorsement from a researcher who has published in `cs.LG` on arXiv. Below are templates you can customize. Send to one person; if they say no or don't reply within a week, try the next.

---

## Who to email (prioritized)

These researchers are known to be responsive to first-time mech-interp authors:

1. **Neel Nanda** (Anthropic / TransformerLens creator) — `neelnanda27@gmail.com`. Famously responsive to interp newcomers.
2. **Joseph Bloom** (sae_lens creator) — find on Twitter/email via personal site.
3. **Lee Sharkey** (Apollo Research) — published early SAE work.
4. **Stefan Heimersheim** (Apollo Research).
5. **Lawrence Chan** (METR, ex-Anthropic).

Look up current email addresses on their personal sites or GitHub. Don't email more than one at a time — it looks bad if two researchers both endorse you (or if you flake on one after they accept).

---

## Template 1 — Direct and concise (recommended)

Subject: `arXiv endorsement request for cs.LG — SAE driver/thermometer paper`

```
Hi [First Name],

I'm a solo independent researcher (no prior arXiv publications) writing to ask
for an endorsement to submit a short empirical paper to arXiv cs.LG.

The work characterizes SAE features in Pythia-160M layer 6 across nine
analyses, and demonstrates that two features both labeled "Newlines" by
auto-interp can have completely different causal roles: one has logit weight
0.49 for newline tokens and produces 19 newlines per 30 tokens when
force-activated; the other has logit weight 0.04 and produces zero. The paper
argues for logit weight analysis as a cheap discriminator between driver and
thermometer features and should be standard in SAE evaluations.

Code and result JSONs: https://github.com/OE-GOD/sae-pythia-160m
Draft PDF: [link or attach]

I'd appreciate an endorsement if you find the work appropriate for cs.LG. My
arXiv endorsement code (after you accept) would let me submit; arXiv just needs
your "yes" via their endorsement portal.

Thanks for your time either way.

Best,
Aung Maw
irving46764@gmail.com
github.com/OE-GOD
```

---

## Template 2 — Slightly warmer (if you've engaged with the person publicly)

Subject: `arXiv endorsement request — Pythia-160M SAE characterization`

```
Hi [First Name],

I've been following your work on [specific paper or post they wrote, e.g.,
"the Gated SAE paper" or "your Alignment Forum posts on SAE interpretability"]
and built a small portfolio piece in roughly that direction. I'd be grateful
if you'd consider endorsing me to submit it to arXiv cs.LG.

Short version: I trained a TopK SAE on Pythia-160M layer 6 (16k features,
k=64, $50 compute budget) and ran nine analyses to characterize what its
features actually do. The headline finding is methodological: two features
both labeled "Newlines" by auto-interp have completely different causal roles
when you test them by residual-stream steering. Logit weight analysis
discriminates them cheaply. I think this is a small but useful pattern for
the field, especially as people start using SAE features for safety
monitoring.

This is my first arXiv submission, so I need an endorsement to upload it.

Repo: https://github.com/OE-GOD/sae-pythia-160m
Writeup: [link or attach the PDF]

If you're willing, arXiv handles the rest — they'll send you a one-click
endorsement link. No commitment beyond that.

Either way, thanks for the work you've put into making this field accessible
to newcomers.

Best,
Aung Maw
```

---

## Template 3 — If you've talked to them before (most natural)

Subject: `[anything personal — "Following up on our chat" / "Your suggestion last week" etc.]`

```
Hi [First Name],

I finished the SAE characterization project I mentioned — wrote it up as a
short paper. Would you be willing to endorse me for arXiv cs.LG so I can
submit it?

[1-2 sentences specific to what you discussed]

Repo: https://github.com/OE-GOD/sae-pythia-160m
PDF attached.

Thanks,
Aung
```

---

## What NOT to do

- **Don't write a long email.** Researchers get hundreds. Stay under 200 words.
- **Don't email multiple people at once.** Looks desperate; only one endorsement is needed; if two accept it's awkward.
- **Don't overshare your goals.** Don't mention MATS, job applications, etc. Stay focused on the work.
- **Don't ask them to read the whole paper.** Endorsement just confirms the work is real research in the right category — it's not peer review.
- **Don't follow up within a week.** Wait at least 5-7 business days before sending a polite "just checking in."
- **Don't take rejection personally.** Many researchers don't endorse strangers, even for good work. Move to the next person on your list.

---

## After they say yes

1. They'll get an email from arXiv with a link.
2. They click "endorse" — one click.
3. You can now submit.
4. **Thank them.** A short follow-up email expressing gratitude when the paper is up is appreciated and builds the relationship.

---

## After your paper is up

1. The arXiv link is permanent — share it on Twitter, LinkedIn, Alignment Forum.
2. Update your GitHub README to link the arXiv preprint.
3. In ~2-4 weeks it appears on Google Scholar.
4. Add it to your scholar profile (you may need to claim it manually if Scholar doesn't auto-match your name).

---

## If you can't get an endorsement

Backup options:
- Post on Alignment Forum (instant, no gatekeeping).
- Use SSRN or OSF Preprints (no endorsement needed, but less prestigious in ML).
- Wait until you have any university affiliation (auto-endorses) or any prior arXiv co-authorship.

But for an ML-style portfolio piece, arXiv is by far the best target. Push through the endorsement hurdle if possible.
