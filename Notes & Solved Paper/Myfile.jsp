<%@ page import="csec.MyBean" %>
<html>
<body>
<jsp:useBean id="csea" class="csec.MyBean"></jsp:useBean>
<jsp:setProperty name="csea" property="name" value="BHAVANI"></jsp:setProperty>
<jsp:setProperty name="csea" property="age" value="19"></jsp:setProperty>
Name:
<jsp:getProperty name="csea" property="name"></jsp:getProperty>
<br/>
Age:
<jsp:getProperty name="csea" property="age"></jsp:getProperty>
</body>
</html>
