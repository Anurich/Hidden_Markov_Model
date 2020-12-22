<h2> Hidden Markov Model </h2>
The HMM is based on augmenting the Markov chain. A Markov chainis a model Markov chain that  tells  us  something  about  the  probabilities  of  sequences  of  random  variables, states, each of which can take on values from some set. These sets can be words, ortags, or symbols representing anything, like the weather.  A Markov chain makes avery strong assumption that if we want to predict the future in the sequence, all thatmatters is the current state. The states before the current state have no impact on thefuture except via the current state. It’s as if to predict tomorrow’s weather you couldexamine today’s weather but you weren’t allowed to look at yesterday’s weather.
A hidden Markov model(HMM) allows us to talk about both observed events Hidden Markov model(like words that we see in the input) and hidden events (like part-of-speech tags) that we think of as causal factors in our probabilistic mode.

<h2>Project Overview</h2>
 In this work, I have implemented Forward and Backward Algorithm to solve the HMM problem. Second, is modification of gym file to integrate my own game, By changing STEP and RESET function as well as instead of using predefined action, I use random action which consist of LEFT, RIGHT, UP, DOWN.
 
 <h2>Installion</h2>
 <ul>
  <li>git clone https://github.com/Anurich/Hidden_Markov_Model </li>
  <li> Python == 2.7.5
</ul>
<h2>Project Directory Overview.</h2>
Forward and Backward Algorithm is implemented in <b>hiddenMarkov file</b>.<br/>
Where as gym game integration is present inside <b>HW 3</b> folder.
