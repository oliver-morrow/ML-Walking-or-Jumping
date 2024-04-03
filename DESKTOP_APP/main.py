import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, 
                             QWidget, QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QMessageBox)
from joblib import load
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import joblib
from extract import split_data, window_feature_extract, SMA, plot_features
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np

class CSVClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'ELEC 292 CSV Classifier'
        self.dataProcessed = False  # Flag to track if the CSV has been processed
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setFixedSize(900, 500)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Welcome title
        self.welcome_label = QLabel("Logistic Labeller")
        self.welcome_label.setObjectName("welcomeLabel")  # Set the object name for styling
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.welcome_label)

        self.button = QPushButton("Need Help? Click here!", self)
        self.button.clicked.connect(self.showHelpWindow)
        
        # Create a horizontal layout with the button centered
        help_button_layout = QHBoxLayout()
        help_button_layout.addWidget(self.button, 0, Qt.AlignCenter)  # Center the button in the layout
        self.main_layout.addLayout(help_button_layout)

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

        self.plot_button = QPushButton('Plot Data', self)
        self.plot_button.clicked.connect(self.plotData)
        self.main_layout.addWidget(self.plot_button, 0, Qt.AlignCenter)

        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        # Button for opening and processing CSV
        self.open_button = QPushButton('Open CSV', self)
        self.open_button.clicked.connect(self.openFileNameDialog)
        self.open_button.setObjectName("selectButton")
        self.button_layout.addWidget(self.open_button)

        # Button for saving the modified CSV
        self.save_button = QPushButton('Save Classified CSV', self)
        self.save_button.clicked.connect(self.onSaveButtonClick)
        self.save_button.setObjectName("saveButton")
        self.button_layout.addWidget(self.save_button)

        self.acknowledgements_layout = QLabel("This app was created by Oliver Morrow, Matthew Szalawiga, and Daniel Heron. ELEC 292, W24")
        self.acknowledgements_layout.setAlignment(Qt.AlignCenter)
        self.acknowledgements_layout.setObjectName("acknowledgements")  # Set the object name for styling
        self.main_layout.addWidget(self.acknowledgements_layout)


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
        data = pd.read_csv(csv_file)
        self.showCSVDataInTable(data, self.initial_table_widget)
        data_split = split_data(data)
        sensor_columns = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 
                          'Acceleration z (m/s^2)', 'Absolute acceleration (m/s^2)']
        data_sma = [SMA(window, sensor_columns, window_size=10) for window in data_split]
        data_features = window_feature_extract(data_sma, 'Absolute acceleration (m/s^2)')
        data_features['label'] = np.nan
        scaler = StandardScaler()
        data_normalized = pd.DataFrame(scaler.fit_transform(data_features.drop(columns=['label'])), columns=data_features.drop(columns=['label']).columns)
        model = joblib.load('model.joblib')
        data_features['label'] = model.predict(data_normalized)
        self.modified_data = data_features
        self.showCSVDataInTable(self.modified_data, self.modified_table_widget)
        self.dataProcessed = True


    def showCSVDataInTable(self, data, table_widget):
        table_widget.clear()
        table_widget.setColumnCount(len(data.columns))
        table_widget.setRowCount(len(data.index))
        table_widget.setHorizontalHeaderLabels(data.columns)

        for i, (index, row) in enumerate(data.iterrows()):
            for j, value in enumerate(row):
                table_widget.setItem(i, j, QTableWidgetItem(str(value)))

        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()


    def saveFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.modified_data.to_csv(fileName, index=False)
            print(f"Classified CSV saved as: {fileName}")

    def showHelpWindow(self):
        appInstructions = """
        <h1>App Instructions</h1>
        <p>Follow the steps below to use this app!</p>
        <ol>
            <li>Click the "Open CSV" button to select a CSV file</li>
            <li>Wait for the file to be processed</li>
            <li>Click the "Save Classified CSV" button to save the classified file</li>"""
        QMessageBox.information(self, "Help", appInstructions)

    def plotData(self):
        if not self.dataProcessed:
            QMessageBox.warning(self, "Action Required", "Please process a CSV file before plotting.")
            return
        
        # Assuming self.modified_data contains the data you want to plot
        dialog = QDialog(self)
        dialog.setWindowTitle("Data Plot")
        dialog.setFixedSize(600, 400)
        layout = QVBoxLayout()

        fig = Figure()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)
        # Example of plotting first two columns against each other
        # Modify according to your data structure
        ax.scatter(self.modified_data.iloc[:, 0], self.modified_data.iloc[:, 1])
        ax.set_title('Plot Title')
        ax.set_xlabel('X-axis Label')
        ax.set_ylabel('Y-axis Label')

        dialog.setLayout(layout)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CSVClassifierApp()
    ex.show()
    sys.exit(app.exec_())
