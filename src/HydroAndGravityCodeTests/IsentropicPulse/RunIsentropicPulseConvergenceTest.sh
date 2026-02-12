#!/bin/bash

echo "Running Isentropic Pulse convergence test..."

julia --project=. RunIsentropicPulseAll.jl

echo "Plotting results..."

python3 PlotIsentropicPulse.py