import glob
import os
import torch
import numpy as np
import time
from load_lib import load_alphazero

alphazero = load_alphazero()

THINK_TIME = 9.5
CPUCT = 1.25
START_TEMP = 1
END_TEMP = 0.2
TEMP_DECAY_HALF_LIFE = 10

Game = alphazero.TakGS4

def calc_temp(turn):
    ln2 = 0.693
    ld = ln2 / TEMP_DECAY_HALF_LIFE
    temp = START_TEMP - END_TEMP
    temp *= np.exp(-ld * turn)
    temp += END_TEMP
    return temp

def eval_position(gs, agent):
    mcts = alphazero.MCTS(CPUCT, gs.num_players(), gs.num_moves(), 0, 1.4, 0.25)
    start = time.time()
    sims = 0
    while time.time() - start < THINK_TIME:
        leaf = mcts.find_leaf(gs)
        v, pi = agent.predict(torch.from_numpy(leaf.canonicalized()))
        v = v.cpu().numpy()
        pi = pi.cpu().numpy()
        mcts.process_result(gs, v, pi, False)
        sims += 1
    print(f"\tRan {sims} simulations in {round(time.time() - start, 3)} seconds")
    
    v, pi = agent.predict(torch.from_numpy(gs.canonicalized()))
    v = v.cpu().numpy()
    pi = pi.cpu().numpy()
    print(f"\tRaw Score: {v}")
    print(f"\tMCTS Current Player WLD: {mcts.root_value()}")
    
    probs = mcts.probs(calc_temp(gs.current_turn()))
    rand = np.random.choice(probs.shape[0], p=probs)
    print(f"\tMCTS Selected Move: {rand}")
    return rand

# PTN parsing is now handled by C++ - no need for Python PTN parser class

def get_player_choice():
    """Get player's choice for who goes first"""
    while True:
        try:
            print("\nWho should go first?")
            print("1. Human (you)")
            print("2. AI")
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == '1':
                return 'human'
            elif choice == '2':
                return 'ai'
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            exit()

def get_board_size():
    """Get board size from user"""
    while True:
        try:
            size = int(input("Enter board size (4-6): ").strip())
            if 4 <= size <= 6:
                return size
            else:
                print("Board size must be between 4 and 6.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            exit()

def main():
    import neural_net
    
    np.set_printoptions(precision=3, suppress=True)
    
    # Get game configuration
    board_size = get_board_size()
    first_player = get_player_choice()
    
    # Load neural network
    nn_folder = os.path.join("data", "checkpoint")
    nn_file = os.path.basename(sorted(glob.glob(os.path.join(nn_folder, "*.pt")))[-1])
    
    print(f"\nUsing network: {nn_file}")
    print(f"Board size: {board_size}x{board_size}")
    print(f"First player: {first_player}")
    
    # Map board size to appropriate TakGS class
    game_classes = {4: alphazero.TakGS4, 5: alphazero.TakGS5, 6: alphazero.TakGS6}
    GameClass = game_classes.get(board_size, alphazero.TakGS4)
    
    nn = neural_net.NNWrapper.load_checkpoint(GameClass, nn_folder, nn_file)
    gs = GameClass()
    
    # Determine if human is player 0 or 1
    human_is_player_0 = (first_player == 'human')
    
    hist = []
    
    while gs.scores() is None:
        hist.append(gs.copy())
        print("\n" + "="*50)
        print(gs)
        print(f"Current player: {gs.current_player()}")
        print(f"Turn: {gs.current_turn()}")
        
        current_is_human = (gs.current_player() == 0 and human_is_player_0) or \
                          (gs.current_player() == 1 and not human_is_player_0)
        
        if current_is_human:
            # Human player's turn
            print("\nYour turn! Enter move in PTN notation (or 'help' for examples, 'undo' to undo):")
            
            valid_move = False
            while not valid_move:
                try:
                    user_input = input("PTN move: ").strip()
                    
                    if user_input.lower() == 'help':
                        print("\nPTN Move Examples:")
                        print("  Flat stone placement: a1, b2, c3")
                        print("  Wall placement: Sa1, Sb2")
                        print("  Capstone placement: Ca1, Cb2")
                        print("  Stone movement: a1>, 2b2<, 3c3+")
                        print("  Complex movement: 4d4<211")
                        continue
                    
                    if user_input.lower() == 'undo':
                        if len(hist) >= 2:
                            gs = hist[-2].copy()
                            hist = hist[:-2]
                            print("Move undone.")
                            break
                        else:
                            print("Cannot undo further.")
                            continue
                    
                    # Try to parse and execute the move using C++ PTN function
                    move = gs.ptn_to_move_index(user_input)
                    valids = gs.valid_moves()
                    
                    if valids[move]:
                        gs.play_move(move)
                        valid_move = True
                        print(f"Played move: {user_input}")
                    else:
                        print(f"Invalid move: {user_input}")
                        
                except ValueError as e:
                    print(f"Error: {e}")
                    print("Please try again or type 'help' for examples.")
                except KeyboardInterrupt:
                    exit()
        
        else:
            # AI player's turn
            print("\nAI thinking...")
            ai_move = eval_position(gs, nn)
            gs.play_move(ai_move)
            print(f"AI played move: {ai_move}")
    
    print("\n" + "="*50)
    print("GAME OVER")
    print(gs)
    print(f"Final scores: {gs.scores()}")
    
    # Determine winner
    scores = gs.scores()
    if scores[0] > scores[1]:
        winner = "Player 0" if not human_is_player_0 else "You"
    elif scores[1] > scores[0]:
        winner = "Player 1" if human_is_player_0 else "You"
    else:
        winner = "Draw"
    
    print(f"Winner: {winner}")

if __name__ == "__main__":
    main()