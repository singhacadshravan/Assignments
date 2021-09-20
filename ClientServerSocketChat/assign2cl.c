#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>

void* receivemessage(void *);
void exitprocess();
int checkmessageformat(char *);
int sockfd;

int main(int argc, char *argv[])
{
    char tinput[250],recv[5];
    int port=2000;
    char server[200]="localhost";
    struct sockaddr_in serveraddr;
    struct hostent *haddr;
    pthread_t rThread;
    signal(SIGINT, exitprocess);	
    if((sockfd=socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("Client socket creation error:");
        exit(-1);
    }
    memset(&serveraddr,'\0', sizeof(struct sockaddr_in));
    if(argc > 1)
    {   
        memset(server,'\0',20);
        strcpy(server,argv[1]);    
    }
    if(argc > 2)
        port=atoi(argv[2]);        
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_port = htons(port);
    haddr=gethostbyname(server);
    if(haddr== NULL)
    {
        perror("gethostbyname()");
        exit(-1);
    }
    serveraddr.sin_addr = *((struct in_addr *)haddr->h_addr);    
    // set Null in rest of the struct
    memset(&(serveraddr.sin_zero), '\0', sizeof serveraddr.sin_zero);
    if(connect(sockfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) == -1)
    {
        perror("connect() system call:");
        exit(-1);
    }
    int status=pthread_create(&rThread,NULL,receivemessage,(void *)sockfd);
	if (status) {
		perror("ERROR: pthread_create():");
		exit(1);
	}
    while(1)
    {
    	fgets(tinput,250,stdin);
    	if(checkmessageformat(tinput))
        	write(sockfd,tinput,strlen(tinput));
        memset(tinput,'\0',250);
        //while(getchar()!='\n');	
    }    
}
int checkmessageformat(char *text)
{
    char *pointer;    
    if(strlen(text)>220)
    {
        printf("Message length cannot exceed 200!\n");        
        return 0;
    }    
    else return 1;
}
/**
 * Takes the input string and verifies the input string for correct format  
 * 
 * @param void* 	
 @return the input string
*/

void* receivemessage(void *socket)
{
    int sockfd,status;
	char buffer[500]; 
	sockfd=(int)socket;
	memset(buffer,'\0',500);
	while(1)
	{
		status=recv(sockfd,buffer,500,0);  
		if (status< 0)
		{		  
			perror("Error in receiving data:\n");    
		}
		else if(status!=0)
		{
			printf("Receiving:");
			fputs(buffer,stdout);
			fflush(stdout);
		}
		else{}
 	}
 }
void exitprocess()
{ 	
    write(sockfd,"KILL",5);
	close(sockfd);  
	exit(1);	
}
