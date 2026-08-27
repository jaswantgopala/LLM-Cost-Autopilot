# LLM Cost Autopilot

Intelligent request routing system that classifies prompt complexity and routes each request to the cheapest model capable of handling it well — with automated quality verification and escalation as a safety net.

## The Headline Number

**93.2% cost reduction** versus routing every request to GPT-4o, while maintaining a 5.0/5.0 average quality score across all logged requests (verified via LLM-as-judge, zero failed escalations in testing).

*(Note: GPT-4o pricing is used as a reference-only baseline for comparison — it is not called anywhere in this system. See Architecture below for the actual models used.)*

## Architecture