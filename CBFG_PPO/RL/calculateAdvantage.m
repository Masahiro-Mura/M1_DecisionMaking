function buffer = calculateAdvantage(buffer, lastValue, pRL)

N = buffer.count;

gamma = pRL.discountFactor;

for t = 1:N

    if buffer.done(t)

        nextValue = 0;

    elseif t < N

        nextValue = buffer.value(t + 1);

    else

        nextValue = lastValue;

    end

    tdError = ...
        buffer.reward(t) ...
        + gamma * nextValue ...
        - buffer.value(t);

    buffer.advantage(t) = tdError;

    buffer.return(t) = ...
        tdError + buffer.value(t);

end

end