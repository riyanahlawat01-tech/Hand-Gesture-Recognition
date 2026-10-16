import javax.swing.*;

public class SwingDemo {
    public static void main(String[] args) {
        // Feature 1: JFrame — main window
        JFrame frame = new JFrame("My First Swing App");

        // Feature 2: Rich components
        JLabel label = new JLabel("Hello Swing!");
        JButton button = new JButton("Click Me");

        // Feature 3: Layout
        frame.setLayout(new java.awt.FlowLayout());
        frame.add(label);
        frame.add(button);

        // Feature 4: Event handling
        button.addActionListener(e ->
            JOptionPane.showMessageDialog(frame, "Button Clicked!"));

        frame.setSize(300, 200);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}