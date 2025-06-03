using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.Text;
using TMPro;

public class OutLoggerManager : MonoBehaviour
{
    public TMP_InputField usernameInput;
    public TMP_InputField passwordInput;
    public Button loginButton;
    public Button logoutButton;  // nuevo botón

    private UdpClient udpClient;
    private string serverIP = "127.0.0.1";
    private int serverPort = 5060; // Asegúrate que sea el mismo puerto que en Python

    void Start()
    {
        udpClient = new UdpClient();
        loginButton.onClick.AddListener(OnLoginClicked);
        logoutButton.onClick.AddListener(OnLogoutClicked);  // nueva función
    }

    void OnLoginClicked()
    {
        string username = usernameInput.text;
        string message = $"USER:{username}";  // o lo que quieras enviar
        byte[] data = Encoding.UTF8.GetBytes(message);
        udpClient.Send(data, data.Length, serverIP, serverPort);
        Debug.Log("Mensaje de login enviado: " + message);
    }

    void OnLogoutClicked()
    {
        string message = "true";  // como espera tu servidor UDP
        byte[] data = Encoding.UTF8.GetBytes(message);
        udpClient.Send(data, data.Length, serverIP, serverPort);
        Debug.Log("Mensaje de logout enviado: " + message);

        // Opcional: cerrar la app
        //Application.Quit();
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;  // Detiene ejecución en editor
        #else
            Application.Quit();  // Cierra el juego en build
        #endif
    }

    private void OnApplicationQuit()
    {
        udpClient.Close();
    }
}
