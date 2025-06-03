using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UDPReceiver : MonoBehaviour
{
    Thread receiveThread;
    UdpClient udpClient;
    int port = 5005;
    string lastMessage = "";

    ///
    private volatile bool isRunning = true;
    ///

    void Start()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void ReceiveData()
    {
        udpClient = new UdpClient(port);
        /*while (true)
        {
            try
            {
                IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, port);
                byte[] data = udpClient.Receive(ref remoteEP);
                string text = Encoding.UTF8.GetString(data);
                lastMessage = text;
                Debug.Log("Mensaje recibido: " + text);

                // Aquí puedes parsear el mensaje y actuar en consecuencia
                // Por ejemplo: "1,derecha"
                string[] parts = text.Split(',');
                if (parts.Length == 2)
                {
                    int boton = int.Parse(parts[0]);
                    string mano = parts[1];

                    // Ejecutar lógica dentro del hilo principal
                    UnityMainThreadDispatcher.Instance().Enqueue(() => HandleSelection(boton, mano));
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e.ToString());
            }
        }*/
        while (isRunning)
        {
            try
            {
                if (udpClient.Available > 0)
                {
                    IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, port);
                    byte[] data = udpClient.Receive(ref remoteEP);
                    string text = Encoding.UTF8.GetString(data);
                    lastMessage = text;
                    Debug.Log("Mensaje recibido: " + text);

                    // Aquí puedes parsear el mensaje y actuar en consecuencia
                    // Por ejemplo: "1,derecha"
                    string[] parts = text.Split(',');
                    if (parts.Length == 2)
                    {
                        int boton = int.Parse(parts[0]);
                        string mano = parts[1];

                        // Ejecutar lógica dentro del hilo principal
                        UnityMainThreadDispatcher.Instance().Enqueue(() => HandleSelection(boton, mano));
                    }
                }
                else
                {
                    Thread.Sleep(10);
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e.ToString());
            }
        }

    }

    void HandleSelection(int boton, string mano)
    {
        Debug.Log($"Botón {boton} presionado con la mano {mano}");
        // Aquí haces la acción deseada en Unity (cambiar escena, activar UI, etc.)
    }

    /*void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        if (udpClient != null) udpClient.Close();
    }*/
    void OnApplicationQuit()
    {
        isRunning = false;
        receiveThread?.Join();
        udpClient?.Close();
    }

}
