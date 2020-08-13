## Written by Lei Xie 
## Flask_restplus 
## Sqlite3
## World Bank API 
## example url = http://api.worldbank.org/v2/countries/all/indicators/NY.GDP.MKTP.CD?date=2012:2017&format=json&per_page=1000

import sqlite3
import requests
from flask import Flask
from flask_restplus import Api, Resource
from flask_restful import reqparse
import json
import os
import time

app = Flask(__name__)
api = Api(app, title="Data Service Engineering Project", description="There are 6 actions available in this API design")

### functions for interaction with sqlite3 database
def db_operation(dbname, command):
	conn = sqlite3.connect(dbname)
	c = conn.cursor()
	# print(command)
	c.execute(command)
	result = c.fetchall()
	conn.commit()
	conn.close()
	return result

def create_db(dbname):
	if os.path.exists(dbname):
		print(f"Database '{dbname}' initialized or already exists...")
		return False
	else:
		print(f"Initializing database '{dbname}'...")	
		conn = sqlite3.connect(dbname)
		c = conn.cursor()
		c.execute(	'''
					CREATE TABLE Collections(
					id INTEGER PRIMARY KEY,
					uri VARCHAR(100),
					indicator VARCHAR(100),
					indicator_value VARCHAR(100),
					creation_time DATE);
					''')
		
		c.execute(  '''
					CREATE TABLE Entries(
					entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
					record_id INTEGER,
					country VARCHAR(100),
					date VARCHAR(100),
					value VARCHAR(100),
					FOREIGN KEY (record_id) REFERENCES Collections (id));
					''')
		conn.commit()
		conn.close()
		return True			


##   	header			id,      uri 			indicator		indicator_value			creation_time
## collection_query = [(1, '/collections/1', 'NY.GDP.MKTP.CD', 'GDP (current US$)', '2020-03-27T12:21:41Z')]	
##	  header	      entry_id,  record_id,       country,     date,        value
## entry_query      = [(1,         1,      'Arab World',    '2017',   '2586506133266.57'), (2, 1, 'Arab World', '2016', '2510521024647.25'), (3, 1, 'Arab World', '2015', '2557895226891.44')]

def combine_entry_record_to_list(entry_query):
	entry_resp = []
	for entry in entry_query:
		json_entry = {
						"country": f"{entry[2]}",
						"date": int(entry[3]),
						"value": f"{entry[4]}"
					}
		entry_resp.append(json_entry)
	return entry_resp



### Set up parser
parser = api.parser()  # 127.0.0.1:5000/collections
parser.add_argument('indicator_id') # 127.0.0.1:5000/collections?indicator_id=XXXXX
parser.add_argument('order_by')
parser.add_argument('q')


### question 1
### HTTP operation: POST /collections?indicator_id=NY.GDP.MKTP.CD

@api.route('/collections')
@api.response(200, 'OK')
@api.response(201, 'Created')
@api.response(400, 'Bad Request')
@api.response(404, 'Not Found')
class Q1(Resource):
	@api.doc(params = {'indicator_id': 'Indicator'})
	def post(self):
		indicator = parser.parse_args()['indicator_id']  # indicator = NY.GDP.MKTP.CD
		if not indicator:
			return {
						"message": "No indicator_id is given"
					}, 400
		query = db_operation('z3457022.db', f"SELECT * FROM Collections WHERE indicator = '{indicator}';")
		if query:  # query = [(1, '/collections/1', 'NY.GDP.MKTP.CD', 'GDP (current US$)', '2020-03-25 22:35:53')]
			existing_resp = {"message": f"The collection of indicator: '{query[0][2]}' was already imported into databse",
							"id" : f"{query[0][0]}",
							"uri" : f"{query[0][1]}",
							"creation_time" : f"{query[0][4]}",
							}
			return existing_resp, 200
		else:
			url = f"http://api.worldbank.org/v2/countries/all/indicators/{indicator}?date=2012:2017&format=json&per_page=1000"
			resp= requests.get(url)
			response = json.loads(resp.content)
		# For invalid indicator api will response as below
		# response = [{'message': [{'id': '120', 'key': 'Invalid value', 'value': 'The provided parameter value is not valid'}]}]
		if len(response) == 1:
			if response[0]['message'][0]['key'] == 'Invalid value':  
				return {
							"message": f"The provided parameter value '{indicator}' is not valid"
						}, 400
		else:
			if not response[1]: #url: http://api.worldbank.org/v2/countries/all/indicators/DT.AMT.DLTT.CD.OT.AR.1824.US?date=2012:2017&format=json&per_page=1000
				return {   # response = [{"page":0,"pages":0,"per_page":0,"total":0,"sourceid":null,"lastupdated":null},null]
					"message": f"Response value for '{indicator}' is null"
				}, 404
			else:
				data_page = response[1]  #response[0]: {"page":1,"pages":2,"per_page":1000,"total":1584,"sourceid":"2","lastupdated":"2020-03-18"},
				new_id = db_operation('z3457022.db', f'SELECT MAX(id) FROM Collections;')
				if not new_id[0][0]:
					new_id = 1
				else:
					new_id = int(new_id[0][0]) + 1
				uri = f"/collections/{new_id}"
				current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())
				# insert values into TABLE Collections
				insert_command = f"INSERT INTO Collections VALUES ({new_id},'{uri}','{data_page[0]['indicator']['id']}','{data_page[0]['indicator']['value']}','{current_time}');"
				db_operation('z3457022.db', insert_command)
				#insert entries values into TABLE Entries
				# subdata = {    "indicator":{"id":"NY.GDP.MKTP.CD","value":"GDP (current US$)"},"country":{"id":"1A","value":"Arab World"},
				# "countryiso3code":"ARB","date":"2017","value":2586506133266.57,"unit":"","obs_status":"","decimal":0     }
				for subdata in data_page: 
					if subdata['value'] is None: # 
						continue
					subdata['country']['value'] = subdata['country']['value'].replace("'", "''")
					insert_command = f"INSERT INTO Entries VALUES (null, {new_id}, '{subdata['country']['value']}','{subdata['date']}','{subdata['value']}');"
					db_operation('z3457022.db', insert_command)
				# POST reponse to api 
				post_resp = {"uri" : f"{uri}",
							"id" : new_id,
							"creation_time" : f"{current_time}",
							"indicator_id" : f"{indicator}"
							}
				return post_resp, 201
	
	
### question 3
### HTTP operation: GET /collections?order_by={+id,+creation_time,+indicator,-id,-creation_time,-indicator}

	@api.doc(params = {'order_by': 'order'})
	def get(self):
		orders = parser.parse_args()['order_by'] # orders = [+id,+creation_time,+indicator,-id,-creation_time,-indicator]
		if orders is None:
			return {
					"message": f"Invalid order: '{orders}' is given. "
					}, 400		
		orders = orders.split(',') 
		query = db_operation('z3457022.db', f"SELECT * FROM Collections;")
		unsorted_res = []
		if not query:
			return {
					"message": f"Data NOT FOUND in Collections TABLE"
					}, 404		
		for item in query:
			json_item = {
							"id": f"{item[0]}",
							"uri": f"{item[1]}",
							"indicator": f"{item[2]}",
							"indicator_value": f"{item[3]}",
							"creation_time": f"{item[4]}"
						}
			unsorted_res.append(json_item)
		for order in orders:
			if not order:
				return {
						"message": f"Invalid order: '{order}' is given. Please select keyword from ['id','creation_time', 'indicator']"
						}, 400				
			else:
				if order[0] == '+':  # sort data as ASCE
					keyword = order[1:]
					if keyword not in ['id','creation_time', 'indicator']:
						return {
								"message": f"Invalid Keyword: '+{keyword}'. Please select keyword from ['id','creation_time', 'indicator']"
								}, 400
					sorted_res = sorted(unsorted_res, key=lambda x: x[order[1:]], reverse=False)
				elif order[0] == '-':  # sort data as DESC
					keyword = order[1:]
					if keyword not in ['id','creation_time', 'indicator']:
						return {
								"message": f"Invalid Keyword: '-{keyword}'. Please select keyword from ['id','creation_time', 'indicator']"
								}, 400
					sorted_res = sorted(unsorted_res, key=lambda x: x[order[1:]], reverse=True)
				else:
					return {
							"message": f"Invalid Order: '{order}'. Orders must start with '+' or '-'"
							}, 400				
		return sorted_res, 200


### question 2
### HTTP operation: DELETE /collections/{id}

@api.route('/collections/<int:id>')
@api.response(200, 'OK')
@api.response(400, 'Bad Request')
@api.response(404, 'Not Found')
class Q2(Resource):
	# @api.doc(params = {'id': 'int'})
	def delete(self,id):
		if isinstance(id, str):
			return {
				"message": f"id: '{id}' is not valid. Please enter an integer"
			}, 400
		query = db_operation('z3457022.db', f"SELECT * FROM Collections WHERE id = {id};")
		if not query:
			return {
				"message": f"The collections/{id} NOT FOUND in database"
			}, 404
		else:
			db_operation('z3457022.db', f"DELETE FROM Entries WHERE record_id = {id};")
			db_operation('z3457022.db', f"DELETE FROM Collections WHERE id = {id};")
			return {
				"message" : f"The collection {id} was removed from database",
				"id" : f"{id}"
			}, 200


### question 4
### HTTP operation: GET /collections/{id}

	def get(self, id):
		if isinstance(id, str):
			return {
				"message": f"id: '{id}' is not valid. Please enter an integer"
			}, 400		
		collection_query = db_operation('z3457022.db', f"SELECT * FROM Collections WHERE id = {id};")
		if not collection_query:
			return {
				"message": f"The collections/{id} NOT FOUND in database"
			}, 404
		entry_query = db_operation('z3457022.db', f"SELECT * FROM Entries WHERE record_id = {id};")
		if not entry_query:
			return {
				"message": f"The entries/{id} NOT FOUND in database"
			}, 404
		entry_resp = combine_entry_record_to_list(entry_query)
		q4_resp = {
					"id": f"{collection_query[0][0]}",
					"uri": f"{collection_query[0][1]}",
					"indicator": f"{collection_query[0][2]}",
					"indicator_value": f"{collection_query[0][3]}",
					"creation_time": f"{collection_query[0][4]}",
					"entries" : entry_resp			
				}
		return q4_resp, 200


### question 5
### HTTP operation: GET /collections/{id}/{year}/{country}

@api.route('/collections/<int:id>/<int:year>/<string:country>')
@api.response(200, 'OK')
@api.response(400, 'Bad Request')
@api.response(404, 'Not Found')
class Q5(Resource):
	def get(self,id, year, country):
		collection_query = db_operation('z3457022.db', f"SELECT * FROM Collections WHERE id = {id};")
		if not collection_query:
			return {
				"message": f"The collections/{id} NOT FOUND in database"
			}, 404
		entry_query = db_operation('z3457022.db', f"SELECT * FROM Entries WHERE record_id = {id} and date={year} and country = '{country}';")
		if not entry_query:
			return {
				"message": f"No matching result for keyword '{id}, {year}, {country}'"
			}, 404
		entry_resp = combine_entry_record_to_list(entry_query)		
		q5_resp = {
					"id": f"{collection_query[0][0]}",
					"uri": f"{collection_query[0][1]}",
					"indicator": f"{collection_query[0][2]}",
					"country": f"{entry_query[0][2]}",
					"year": int(entry_query[0][3]),
					"value": f"{entry_query[0][4]}"		
				}
		return q5_resp, 200		


### question 6
### GET /collections/{id}/{year}?q=<query>

@api.route('/collections/<int:id>/<int:year>')
@api.response(200, 'OK')
@api.response(400, 'Bad Request')
@api.response(404, 'Not Found')
class Q6(Resource):
	@api.doc(params = {'q': 'query'})
	def get(self, id, year):
		q = parser.parse_args()['q']  # query = +N, N, -N, None
		collection_query = db_operation('z3457022.db', f"SELECT * FROM Collections WHERE id = {id};")
		if not collection_query:
			return {
				"message": f"The collections/{id} NOT FOUND in database"
			}, 404
		entry_query = db_operation('z3457022.db', f"SELECT * FROM Entries WHERE record_id = {id} and date={year};")
		if not entry_query:
			return {
				"message": f"No matching result for keyword '{id}, {year}'"
			}, 404
		entry_resp = q6_combine_entry_record_to_list(entry_query)	
		top_sorted_entry = sorted(entry_resp, key=lambda x: x['value'], reverse=True)	
		if not q:
			return q6_resp_format(collection_query, entry_query, top_sorted_entry), 200					
		if q[0] == '+':
			print(q)
			print(q[1:])
			if not q6_validNumber(q[1:]):
				return {
							"message": f"Invalid query '+{q[1:]}'. Query format: '+N','-N','N' where N ~ (1-100)"
						}, 400				
			quantity = int(q[1:])
			top_sorted_entry = top_sorted_entry[:quantity]	
			return q6_resp_format(collection_query, entry_query, top_sorted_entry), 200			
		elif q[0] == '-':
			if not q6_validNumber(q[1:]):
				return {
							"message": f"Invalid query '-{q[1:]}'. Query format: '+N','-N','N' where N ~ (1-100)"
						}, 400
			quantity = int(q[1:])
			bottom_sorted_entry = top_sorted_entry[::-1]
			bottom_sorted_entry = bottom_sorted_entry[:quantity]	

			return q6_resp_format(collection_query, entry_query, bottom_sorted_entry), 200				
		else:
			if not q6_validNumber(q):
				return {
							"message": f"Invalid query '{q}'. Query format: '+N','-N','N' where N ~ (1-100)"
						}, 400				
			quantity = int(q)
			top_sorted_entry = top_sorted_entry[:quantity]	

			return q6_resp_format(collection_query, entry_query, top_sorted_entry), 200


def q6_validNumber(input):
	try:
		number = int(input)
	except:
		return False
	if 1 <= number <= 100:
		return True
	else:
		return False

def q6_resp_format(collection_query, entry_query, sorted_entry):
	q6_query = {
					"id": f"{collection_query[0][0]}",
					"uri": f"{collection_query[0][1]}",
					"indicator": f"{collection_query[0][2]}",
					"indicator_value": f"{collection_query[0][3]}",
					"creation_time": f"{collection_query[0][4]}",
					"year" : int(entry_query[0][3]),
					"entries" : sorted_entry			
					}	
	return q6_query, 200	
	
def q6_combine_entry_record_to_list(entry_query):
	entry_resp = []
	for entry in entry_query:
		json_entry = {
						"country": f"{entry[2]}",
						"value": float(entry[4])
					}
		entry_resp.append(json_entry)
	return entry_resp




if __name__ == '__main__':
	create_db('z3457022.db')
	# app.run(debug=True)
	try:
		app.run(debug=True)
	except:
		app.run(port=8000,debug=True)
