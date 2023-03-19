from django.db import models

# Create your models here.

class stocks(models.Model):
    #stock -> username    stock    buy_quantity    sell_quantity    buy_amount    sell_amount    buy_time    sell_time    buy_price  sell_price    status(buy/sell)
    #holdings -> username    stock    quantity    buy_amount    avg_buy_price
    username = models.TextField()
    stock = models.TextField()
    buy_quantity = models.FloatField(default=None, null=True, blank=True)
    sell_quantity = models.FloatField(default=None, null=True, blank=True)
    buy_amount = models.FloatField(default=None, null=True, blank=True)
    sell_amount = models.FloatField(default=None, null=True, blank=True)
    buy_time = models.DateTimeField(auto_now_add=True)
    sell_time = models.DateTimeField(default=None, null=True, blank=True)
    buy_price = models.FloatField(default=None, null=True, blank=True)
    sell_price = models.FloatField(default=None, null=True, blank=True)
    status = models.BooleanField()

class holdings(models.Model):
    username = models.TextField()
    stock = models.TextField()
    buy_amount = models.FloatField()
    quantity = models.FloatField()
    avg_buy_price = models.FloatField()