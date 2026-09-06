clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pRL = rlParameters();

buffer = createExperienceBuffer(pRL);

%% Experience 1

buffer.reward(1) = 1.0;
buffer.value(1) = 0.5;
buffer.done(1) = false;

%% Experience 2

buffer.reward(2) = 2.0;
buffer.value(2) = 1.0;
buffer.done(2) = false;

%% Experience 3

buffer.reward(3) = 3.0;
buffer.value(3) = 1.5;
buffer.done(3) = true;

buffer.count = 3;

lastValue = 10.0;

buffer = calculateAdvantage( ...
    buffer, ...
    lastValue, ...
    pRL);

gamma = pRL.discountFactor;

expectedAdvantage1 = ...
    1.0 + gamma * 1.0 - 0.5;

expectedAdvantage2 = ...
    2.0 + gamma * 1.5 - 1.0;

expectedAdvantage3 = ...
    3.0 - 1.5;

expectedReturn1 = ...
    expectedAdvantage1 + 0.5;

expectedReturn2 = ...
    expectedAdvantage2 + 1.0;

expectedReturn3 = ...
    expectedAdvantage3 + 1.5;

assert(abs( ...
    buffer.advantage(1) ...
    - expectedAdvantage1) < 1e-10);

assert(abs( ...
    buffer.advantage(2) ...
    - expectedAdvantage2) < 1e-10);

assert(abs( ...
    buffer.advantage(3) ...
    - expectedAdvantage3) < 1e-10);

assert(abs( ...
    buffer.return(1) ...
    - expectedReturn1) < 1e-10);

assert(abs( ...
    buffer.return(2) ...
    - expectedReturn2) < 1e-10);

assert(abs( ...
    buffer.return(3) ...
    - expectedReturn3) < 1e-10);

disp("Terminal case passed.");

%% Bootstrap case

buffer2 = createExperienceBuffer(pRL);

buffer2.reward(1) = 1.0;
buffer2.value(1) = 0.5;
buffer2.done(1) = false;

buffer2.reward(2) = 2.0;
buffer2.value(2) = 1.0;
buffer2.done(2) = false;

buffer2.count = 2;

lastValue = 2.0;

buffer2 = calculateAdvantage( ...
    buffer2, ...
    lastValue, ...
    pRL);

expectedAdvantage1 = ...
    1.0 + gamma * 1.0 - 0.5;

expectedAdvantage2 = ...
    2.0 + gamma * 2.0 - 1.0;

expectedReturn1 = ...
    expectedAdvantage1 + 0.5;

expectedReturn2 = ...
    expectedAdvantage2 + 1.0;

assert(abs( ...
    buffer2.advantage(1) ...
    - expectedAdvantage1) < 1e-10);

assert(abs( ...
    buffer2.advantage(2) ...
    - expectedAdvantage2) < 1e-10);

assert(abs( ...
    buffer2.return(1) ...
    - expectedReturn1) < 1e-10);

assert(abs( ...
    buffer2.return(2) ...
    - expectedReturn2) < 1e-10);

disp("Bootstrap case passed.");

disp("Advantage calculation test passed.");