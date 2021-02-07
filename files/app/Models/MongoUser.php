<?php

namespace App\Models;

use Jenssegers\Mongodb\Auth\User as Authenticatable;
use Jenssegers\Mongodb\Eloquent\SoftDeletes;

class MongoUser extends Authenticatable
{
    use SoftDeletes;

    protected $collection = 'users';

    protected $primaryKey = 'id';

    protected $dates = ['deleted_at'];

    protected $guarded = [];
}
