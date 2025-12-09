import sys
import os
import time

# Ensure we can import from the main directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'main')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'main', 'models')))

# Now import the models
try:
    from main.models import svm
    from main.models import naive_bayes
    from main.models import rnn_model
    from main.models import bi_lstm
except ImportError as e:
    print(f"Error importing models: {e}")
    # Try alternative import path if running from inside main
    try:
        from models import svm, naive_bayes, rnn_model, bi_lstm
    except ImportError as e2:
        print(f"Critical error: Could not import models. {e2}")
        sys.exit(1)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    print("\n========================================")
    print("      ReDNA Model Evaluation Menu")
    print("========================================")
    print("1. SVM")
    print("2. Naive Bayes")
    print("3. RNN")
    print("4. Bi-LSTM")
    print("5. Test All Models")
    print("0. Exit")
    print("========================================")

def run_evaluation(choice):
    if choice == '1':
        svm.evaluate_model()
    elif choice == '2':
        naive_bayes.evaluate_model()
    elif choice == '3':
        rnn_model.evaluate_model()
    elif choice == '4':
        bi_lstm.evaluate_model()
    elif choice == '5':
        print("\n--- Testing SVM ---")
        svm.evaluate_model()
        print("\n--- Testing Naive Bayes ---")
        naive_bayes.evaluate_model()
        print("\n--- Testing RNN ---")
        rnn_model.evaluate_model()
        print("\n--- Testing Bi-LSTM ---")
        bi_lstm.evaluate_model()
    else:
        print("Invalid choice.")

def main():
    while True:
        print_menu()
        choice = input("Enter your choice: ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        if choice in ['1', '2', '3', '4', '5']:
            try:
                run_evaluation(choice)
            except Exception as e:
                print(f"An error occurred during evaluation: {e}")
            
            input("\nPress Enter to return to menu...")
        else:
            print("Invalid selection, please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
