# LSTM_Stacked_1.0

Es gibt zwei Hauptprogramme.
Einmal Das LSTMSinusTest.py für die einfache LSTM Zelle und das LSTMSinusTestStacked für die Stacked Architekur.
Das Monitoring der Gates ist nur bei der einzelnen Zelle möglich. Ist das PRogramm LSTMSinusTest durchgelaufen,
muss das Programm LSTMMonitor gestartet werden um die Gates zu visualisieren

Für die einfache LSTM Zelle wird die Backpropagation von der Klasse LSTMBackPropagator.py genutzt und
für die Stacked Architektur die Klasse LSTMStackedBAckpropagator.py

Die beiden Testklassen wurden dafür genutzt die Backpropagation und die LSTM Zelle selbst zu testen,
indem vordefinierte  Werte in die Gates eingegeben wurde und diese mit der Berechnungen verglichen
