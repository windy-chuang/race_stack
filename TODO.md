# TODO

1. Compare IBR trajectory when there is opponent vs. no opponent (when opponent is close vs. far away)

2. Visualise planning trajectory

3. Implement the actual paper (explain the difference to microsoft's code)

4. How is the performance of this compared to a simple rule-based defender? Or something else? Like detecting trailing, speeding-up, and changing lines 

5. Understand Pure Pursuit (PP)

6. We could actually take something from SGTP

7. Alpha-Racer

8. Verify whether IBR actually converges and (theory) show that NE is achieved

9. Understand DPP

10. Was overtaken in glc_ot_ez

11. Could I work with vision pipeline?

12. Consider opponent's speed

13. Print the value function

14. Read the car paper (Wang)

15. Warm-start?

16. Should we measure the progress with centerline?

17. Look-ahead distance

18. Find out how many loops we could actually do (also on the real car)

19. Implement and compare with exploitation of specific overtaking algorithm

20. Read spliner code (speed up, max speed etc.)

21. Maximising progress along gb?

22. Compare to MPC

23. Fast and slow IBRs

24. Import different maps (wide margin etc.)

























# Progress this week

1. Added blocking term
2. Tuned collision distance
3. SGTP actually uses control!
4. Fast spliner vs. slow gb: 0.6 could still overtake, 0.7 didn't succeed after several rounds

