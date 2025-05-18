# TESTING FILE
import tkinter as tk
from tkinter import ttk

# Main application window
root = tk.Tk()
root.title("Facial Expression Detector")
root.geometry("740x480")
root.resizable(False,False)

# Function to switch frames
def show_page(page):
    # Hide all frames
    for widget in root.winfo_children():
        widget.grid_forget()

    # Show the new page
    page.grid(row=0, column=0, sticky="nsew")

# Main Menu
menu1 = tk.Menu(root)

# Sub-menu for 'Statistics'
file_menu = tk.Menu(menu1, tearoff=False)
file_menu.add_command(label="Home", command=lambda: show_page(home_page))
file_menu.add_command(label="Statistics", command=lambda: show_page(page2))
file_menu.add_command(label="test", command=lambda: show_page(page3))
menu1.add_cascade(label='Menu', menu=file_menu)

root.configure(menu=menu1)

# First Frame (Home Page)
home_page = tk.Frame(root)
home_label = tk.Label(home_page, text="Welcome to the Facial Expression Detector!", font=("Arial", 16),justify="center")
home_label.grid(row=0, column=0, padx=20, pady=20)

# Second Frame (New Page that appears when you select Command)
page2 = tk.Frame(root)
page2_label = tk.Label(page2, text="This is Page 2", font=("Arial", 16))
page2_label.grid(row=0, column=0, padx=20, pady=20)

# test
page3 = tk.Frame(root)
page3_label = tk.Label(page3, text="Line 1\nSecond line")
page3_label.pack()

# Set the home page as the default
show_page(home_page)

# Run the application
root.mainloop()
