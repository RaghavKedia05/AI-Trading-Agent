# -----------------------------
# Run AI Agent
# -----------------------------

if run_ai:

    st.subheader("🤖 AI Trading Signal")

    try:

        env = TradingEnvironment(data)

        agent = DQNAgent(
            state_size=4,
            action_size=3
        )

        # Disable randomness so results are consistent
        agent.epsilon = 0

        state = env.reset()

        action = agent.act(state)

        actions = {
            0: "HOLD",
            1: "BUY",
            2: "SELL"
        }

        signal = actions[action]

        if signal == "BUY":
            st.success("🟢 BUY SIGNAL")

        elif signal == "SELL":
            st.error("🔴 SELL SIGNAL")

        else:
            st.warning("🟡 HOLD POSITION")

    except Exception as e:
        st.error("AI model failed to run. Try another stock.")
