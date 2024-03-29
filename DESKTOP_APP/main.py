import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QWidget, QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QMessageBox)
from joblib import load

class CSVClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'CSV Classifier'
        self.dataProcessed = False  # Flag to track if the CSV has been processed
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Welcome paragraph
        self.welcome_label = QLabel("Welcome to the CSV Classifier. Please open a CSV file to begin.")
        self.welcome_label.setObjectName("welcomeLabel")  # Set the object name for styling
        self.main_layout.addWidget(self.welcome_label)
        
        # Horizontal layout for tables
        self.tables_layout = QHBoxLayout()
        self.main_layout.addLayout(self.tables_layout)

        # Table for displaying initial CSV content
        self.initial_table_widget = QTableWidget()
        self.initial_table_widget.setMaximumSize(600, 400)
        self.tables_layout.addWidget(self.initial_table_widget)

        # Table for displaying modified CSV content
        self.modified_table_widget = QTableWidget()
        self.modified_table_widget.setMaximumSize(600, 400)
        self.tables_layout.addWidget(self.modified_table_widget)

        # Button for opening and processing CSV
        self.open_button = QPushButton('Open CSV', self)
        self.open_button.clicked.connect(self.openFileNameDialog)
        self.main_layout.addWidget(self.open_button)

        # Button for saving the modified CSV
        self.save_button = QPushButton('Save Classified CSV', self)
        self.save_button.clicked.connect(self.onSaveButtonClick)
        self.main_layout.addWidget(self.save_button)

        self.loadStyles()
    
    def loadStyles(self):
            with open('styles.css', 'r') as f:
                self.setStyleSheet(f.read())

    def onSaveButtonClick(self):
        if not self.dataProcessed:
            QMessageBox.warning(self, "Action Required", "Please process a CSV file before saving.")
        else:
            self.saveFileDialog()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.classifyCSV(fileName)

    def classifyCSV(self, csv_file):
        model = load('logistic_regression_model.joblib')
        data = pd.read_csv(csv_file)
        predictions = model.predict(data)
        data['Prediction'] = predictions
        self.showCSVDataInTable(data)
        self.dataProcessed = True

    def showCSVDataInTable(self, data):
        self.modified_table_widget.clear()
        self.modified_table_widget.setColumnCount(len(data.columns))
        self.modified_table_widget.setRowCount(len(data.index))
        self.modified_table_widget.setHorizontalHeaderLabels(data.columns)
        
        for i, (index, row) in enumerate(data.iterrows()):
            for j, value in enumerate(row):
                self.modified_table_widget.setItem(i, j, QTableWidgetItem(str(value)))
                
        self.modified_table_widget.resizeColumnsToContents()
        self.modified_table_widget.resizeRowsToContents()

    def saveFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.modified_data.to_csv(fileName, index=False)
            print(f"Classified CSV saved as: {fileName}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CSVClassifierApp()
    ex.show()
    sys.exit(app.exec_())
