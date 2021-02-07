<?php

namespace App\Models;

use Jenssegers\Mongodb\Eloquent\Model;

class UserDocument extends Model
{
    protected $collection = 'user_documents';

    protected $guarded = [];
}
